"""
Audio Quality Analyzer (backend/app/ai/audio/quality.py)
--------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Dataclasses (`@dataclass`): Immutable structured representation of audio metrics.
- Vectorized DSP Analysis: Efficient computation of SNR (dB), clipping ratio,
  noise floor, RMS energy, and spectral flatness using NumPy & SciPy.
- Rule-based Heuristics: Generates actionable plain-language feedback for voice
  enrollment recordings.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
from scipy import signal
from scipy.stats import gmean

from app.ai.audio.vad import vad_processor
from app.core.logging import logger, timed_step


@dataclass
class AudioQualityReport:
    """
    Standardized audio telemetry report evaluating microphone capture quality
    prior to voice cloning enrollment.
    """
    duration_seconds: float
    speech_duration_seconds: float
    speech_ratio: float
    snr_db: float
    clipping_ratio: float
    noise_floor_db: float
    peak_dbfs: float
    spectral_flatness: float
    is_usable: bool
    recommendation: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AudioQualityAnalyzer:
    """
    Analyzes raw or processed voice recordings, assessing Signal-to-Noise Ratio (SNR),
    clipping distortion, voice activity ratio, and frequency balance.
    """

    @timed_step("Audio Quality Analysis")
    def analyze(self, audio: np.ndarray, sr: int) -> AudioQualityReport:
        """
        Runs comprehensive acoustic evaluation on a float32 1D audio array.
        """
        if len(audio) == 0:
            return AudioQualityReport(
                duration_seconds=0.0,
                speech_duration_seconds=0.0,
                speech_ratio=0.0,
                snr_db=0.0,
                clipping_ratio=0.0,
                noise_floor_db=-100.0,
                peak_dbfs=-100.0,
                spectral_flatness=0.0,
                is_usable=False,
                recommendation="No audio data detected."
            )

        # 1. Total Duration & Peak Amplitude
        duration = len(audio) / sr
        peak_amp = float(np.max(np.abs(audio)))
        peak_dbfs = 20.0 * np.log10(max(peak_amp, 1e-6))

        # 2. Clipping Ratio (samples reaching or exceeding 0.999)
        clipped_samples = int(np.sum(np.abs(audio) >= 0.999))
        clipping_ratio = clipped_samples / len(audio)

        # 3. Speech Segments via VAD
        timestamps = vad_processor.get_speech_timestamps(audio, sr)
        total_speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        speech_duration = total_speech_samples / sr
        speech_ratio = speech_duration / duration if duration > 0 else 0.0

        # 4. Energy Estimation for SNR and Noise Floor
        frame_len = int(sr * 0.05)  # 50 ms analysis window
        hop_len = int(sr * 0.025)   # 25 ms hop
        num_frames = (len(audio) - frame_len) // hop_len + 1

        if num_frames > 0:
            frame_rms = np.array([
                np.sqrt(np.mean(audio[i * hop_len : i * hop_len + frame_len] ** 2) + 1e-12)
                for i in range(num_frames)
            ], dtype=np.float32)

            # Noise floor estimated from 10th percentile of quietest frames
            noise_floor_rms = float(np.percentile(frame_rms, 10))
            noise_floor_db = 20.0 * np.log10(max(noise_floor_rms, 1e-6))

            # Speech energy estimated from 85th percentile of loud frames
            speech_rms = float(np.percentile(frame_rms, 85))
            speech_db = 20.0 * np.log10(max(speech_rms, 1e-6))

            snr_db = max(0.0, speech_db - noise_floor_db)
        else:
            noise_floor_db = -60.0
            snr_db = 20.0

        # 5. Spectral Flatness (Wiener entropy)
        # Ratio of geometric mean to arithmetic mean of power spectrum
        try:
            freqs, psd = signal.welch(audio, sr, nperseg=min(len(audio), 2048))
            psd = psd + 1e-12  # avoid log(0)
            flatness = float(gmean(psd) / np.mean(psd))
        except Exception:
            flatness = 0.1

        # 6. Actionable Recommendation Heuristics
        recommendation, is_usable = self._generate_recommendation(
            duration=duration,
            speech_duration=speech_duration,
            speech_ratio=speech_ratio,
            snr_db=snr_db,
            clipping_ratio=clipping_ratio,
            peak_dbfs=peak_dbfs,
            noise_floor_db=noise_floor_db
        )

        return AudioQualityReport(
            duration_seconds=round(duration, 2),
            speech_duration_seconds=round(speech_duration, 2),
            speech_ratio=round(speech_ratio, 3),
            snr_db=round(snr_db, 1),
            clipping_ratio=round(clipping_ratio, 5),
            noise_floor_db=round(noise_floor_db, 1),
            peak_dbfs=round(peak_dbfs, 1),
            spectral_flatness=round(flatness, 4),
            is_usable=is_usable,
            recommendation=recommendation
        )

    def _generate_recommendation(
        self,
        duration: float,
        speech_duration: float,
        speech_ratio: float,
        snr_db: float,
        clipping_ratio: float,
        peak_dbfs: float,
        noise_floor_db: float
    ) -> tuple[str, bool]:
        """Generates plain-language advice for improving voice recordings."""
        recs: List[str] = []
        is_usable = True

        if clipping_ratio > 0.001 or peak_dbfs > -0.1:
            recs.append("Audio is clipping (distorting). Turn down microphone gain.")
            is_usable = False

        if peak_dbfs < -22.0:
            recs.append("Recording volume is too low. Move closer to the microphone.")

        if speech_duration < 8.0:
            recs.append(f"Only {speech_duration:.1f}s of speech detected. Provide 15-30s for rich timbre.")
            is_usable = False

        if snr_db < 12.0 or noise_floor_db > -42.0:
            recs.append("Excessive background noise detected. Record in a quiet room.")
            if snr_db < 8.0:
                is_usable = False

        if speech_ratio < 0.35 and duration > 10.0:
            recs.append("Excessive silence pauses detected. Speak continuously during enrollment.")

        if not recs:
            recommendation = "Studio quality voice sample! Clean dynamics and high signal-to-noise ratio."
            is_usable = True
        else:
            recommendation = " | ".join(recs)

        return recommendation, is_usable


quality_analyzer = AudioQualityAnalyzer()
