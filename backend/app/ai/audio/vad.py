"""
Voice Activity Detection (VAD) Module (backend/app/ai/audio/vad.py)
-------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Singleton / Lazy-loading Pattern: Silero VAD model is loaded into memory once on
  demand, avoiding redundant cold-start latencies.
- NumPy Vectorized Signal Processing: Resampling, frame chunking, and sample indexing
  performed in fast C/Fortran primitives without Python-level loops.
- Fallback Strategy Pattern: If deep neural VAD fails on an unusual signal,
  an energy-based VAD (short-time energy + zero-crossing rate) gracefully handles the stream.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from scipy import signal

from app.core.logging import logger, timed_step


class SileroVADProcessor:
    """
    Manages Silero VAD model inference for speech boundary detection,
    leading/trailing silence trimming, and internal pause capping.
    """
    _instance: Optional["SileroVADProcessor"] = None
    _model = None
    _utils = None

    def __new__(cls) -> "SileroVADProcessor":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _ensure_loaded(self) -> None:
        """Lazily load Silero VAD neural network on CPU."""
        if self._model is None:
            logger.info("Loading Silero VAD neural network into CPU memory...")
            try:
                # Load from torch.hub with local caching
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad",
                    model="silero_vad",
                    trust_repo=True,
                    verbose=False
                )
                self._model = model
                self._utils = utils
                logger.info("Silero VAD loaded successfully.")
            except Exception as exc:
                logger.warning(f"Could not load Silero VAD from torch.hub ({exc}). Energy fallback will be used.")

    def _resample(self, audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Vectorized polyphase resampler using scipy.signal.resample_poly."""
        if orig_sr == target_sr:
            return audio
        gcd = np.gcd(orig_sr, target_sr)
        up = target_sr // gcd
        down = orig_sr // gcd
        return signal.resample_poly(audio, up, down).astype(np.float32)

    def is_speech(
        self,
        chunk: np.ndarray,
        sr: int,
        threshold: float = 0.35
    ) -> bool:
        """
        Real-time frame-level speech activity decision.
        Returns True if the chunk contains active human speech above the probability threshold.
        """
        if len(chunk) == 0:
            return False

        # 1. Try neural Silero VAD
        self._ensure_loaded()
        if self._model is not None:
            try:
                target_sr = 16000
                audio_16k = self._resample(chunk, sr, target_sr) if sr != target_sr else chunk

                # Silero requires min 512 samples at 16k
                if len(audio_16k) >= 512:
                    tensor = torch.from_numpy(audio_16k[:1536].astype(np.float32))
                    speech_prob = self._model(tensor, target_sr).item()
                    return bool(speech_prob >= threshold)
            except Exception:
                pass

        # 2. Fast Energy / RMS fallback
        rms = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2) + 1e-12))
        return rms > 0.015

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        sr: int,
        threshold: float = 0.35,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 150
    ) -> List[Dict[str, int]]:
        """
        Detects start and end sample indices of active speech segments.
        Returns a list of dicts: [{'start': int, 'end': int}, ...] in native sample units.
        """
        self._ensure_loaded()

        if len(audio) == 0:
            return []

        # Silero VAD expects 16,000 Hz or 8,000 Hz float32 mono
        target_sr = 16000
        if sr != target_sr:
            audio_16k = self._resample(audio, sr, target_sr)
        else:
            audio_16k = audio

        timestamps_native: List[Dict[str, int]] = []

        if self._model is not None and self._utils is not None:
            try:
                get_timestamps_fn = self._utils[0]
                tensor_16k = torch.from_numpy(audio_16k.astype(np.float32))
                timestamps_16k = get_timestamps_fn(
                    tensor_16k,
                    self._model,
                    sampling_rate=target_sr,
                    threshold=threshold,
                    min_speech_duration_ms=min_speech_duration_ms,
                    min_silence_duration_ms=min_silence_duration_ms
                )

                # Scale timestamp indices back to original sample rate
                ratio = sr / target_sr
                for item in timestamps_16k:
                    start_orig = int(round(item["start"] * ratio))
                    end_orig = min(int(round(item["end"] * ratio)), len(audio))
                    if end_orig > start_orig:
                        timestamps_native.append({"start": start_orig, "end": end_orig})
                
                if timestamps_native:
                    return timestamps_native
            except Exception as e:
                logger.debug(f"Silero VAD timestamp detection encountered: {e}. Falling back to energy VAD.")

        # Energy-based VAD Fallback
        return self._energy_vad(audio, sr, min_speech_duration_ms, min_silence_duration_ms)

    def _energy_vad(
        self,
        audio: np.ndarray,
        sr: int,
        min_speech_duration_ms: int = 200,
        min_silence_duration_ms: int = 150
    ) -> List[Dict[str, int]]:
        """
        Robust short-time energy (STE) VAD fallback.
        Computes RMS energy across 20ms frames with adaptive noise thresholding.
        """
        frame_len = int(sr * 0.02)  # 20ms
        hop_len = int(sr * 0.01)    # 10ms

        if len(audio) < frame_len:
            return [{"start": 0, "end": len(audio)}]

        # Calculate short-time RMS energy
        num_frames = (len(audio) - frame_len) // hop_len + 1
        rms = np.zeros(num_frames, dtype=np.float32)
        for i in range(num_frames):
            frame = audio[i * hop_len : i * hop_len + frame_len]
            rms[i] = np.sqrt(np.mean(frame ** 2) + 1e-12)

        # Adaptive threshold: 10th percentile (noise floor) + dynamic range fraction
        noise_floor = np.percentile(rms, 15)
        peak_energy = np.percentile(rms, 95)
        thresh = noise_floor + 0.08 * (peak_energy - noise_floor)
        thresh = max(thresh, 0.005)

        is_speech = rms > thresh
        min_speech_frames = max(1, min_speech_duration_ms // 10)
        min_silence_frames = max(1, min_silence_duration_ms // 10)

        # Merge short silence dips and remove short bursts
        segments: List[Dict[str, int]] = []
        in_speech = False
        start_frame = 0
        silence_count = 0

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_speech:
                    in_speech = True
                    start_frame = i
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_frames:
                        end_frame = i - silence_count
                        if (end_frame - start_frame) >= min_speech_frames:
                            start_sample = start_frame * hop_len
                            end_sample = min(len(audio), end_frame * hop_len + frame_len)
                            segments.append({"start": start_sample, "end": end_sample})
                        in_speech = False
                        silence_count = 0

        if in_speech and (len(is_speech) - start_frame) >= min_speech_frames:
            segments.append({
                "start": start_frame * hop_len,
                "end": len(audio)
            })

        return segments

    @timed_step("VAD Silence Trimming & Pause Capping")
    def trim_silence(
        self,
        audio: np.ndarray,
        sr: int,
        max_internal_pause_sec: float = 0.7,
        padding_sec: float = 0.1
    ) -> np.ndarray:
        """
        1. Trims leading and trailing silence outside active speech boundaries.
        2. Caps internal pauses to `max_internal_pause_sec` so speech flows naturally
           without unnatural dead air.
        """
        timestamps = self.get_speech_timestamps(audio, sr)
        if not timestamps:
            # If no speech was detected, return original audio
            return audio

        padding_samples = int(padding_sec * sr)
        max_pause_samples = int(max_internal_pause_sec * sr)

        chunks: List[np.ndarray] = []

        for i, segment in enumerate(timestamps):
            start = max(0, segment["start"] - padding_samples)
            end = min(len(audio), segment["end"] + padding_samples)

            # Insert capped pause between consecutive segments
            if i > 0:
                prev_end = min(len(audio), timestamps[i - 1]["end"] + padding_samples)
                gap_len = start - prev_end
                if gap_len > 0:
                    pause_to_keep = min(gap_len, max_pause_samples)
                    pause_chunk = audio[prev_end : prev_end + pause_to_keep]
                    chunks.append(pause_chunk)

            speech_chunk = audio[start:end]
            chunks.append(speech_chunk)

        if not chunks:
            return audio

        return np.concatenate(chunks).astype(np.float32)


# Global singleton instance
vad_processor = SileroVADProcessor()
