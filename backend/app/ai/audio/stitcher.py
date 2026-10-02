"""
Audio Stitcher with Equal-Power Crossfading (backend/app/ai/audio/stitcher.py)
-----------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Equal-Power Crossfading: Uses trigonometric sinusoidal curves (cos/sin) where
  fade_out^2 + fade_in^2 == 1.0, preserving constant sound power across transitions
  without dips in volume or phase cancellation.
- Calibrated Room Tone Insertion: Fills inter-sentence pauses with subtle -65 dBFS
  shaped noise, avoiding the artificial "digital silence" sensation.
- Vectorized Audio Concatenation: Efficient NumPy slicing and blending for long-form
  speech assemblies (10-minute to 1-hour audiobooks and podcasts).
"""

from dataclasses import dataclass
from typing import List, Optional
import numpy as np
from scipy import signal

from app.core.logging import logger, timed_step


@dataclass
class StitcherConfig:
    """Configuration parameters for seamless speech stitching."""
    crossfade_ms: float = 30.0         # 20ms - 40ms equal-power crossfade
    fade_boundary_ms: float = 15.0     # Smooth fade in/out for pause boundaries
    insert_room_tone: bool = True      # Insert faint room tone during pauses
    room_tone_dbfs: float = -65.0      # -65 dBFS noise floor
    target_sample_rate: int = 24_000   # Default TTS sample rate


class AudioStitcher:
    """
    Assembles synthesized audio chunks into a studio-grade continuous stream.
    Applies equal-power crossfades and natural pause room tone.
    """

    def __init__(self, config: Optional[StitcherConfig] = None):
        self.config = config or StitcherConfig()

    def generate_room_tone(self, duration_sec: float, sr: int) -> np.ndarray:
        """
        Generates subtle, low-pass filtered room tone at target dBFS.
        Simulates natural acoustic studio room tone rather than harsh white noise.
        """
        samples = int(duration_sec * sr)
        if samples <= 0:
            return np.zeros(0, dtype=np.float32)

        # Base gaussian noise at target amplitude
        amp = 10.0 ** (self.config.room_tone_dbfs / 20.0)
        noise = np.random.normal(0, amp, size=samples).astype(np.float32)

        # Gentle 1st-order low-pass filter at 1200 Hz to simulate soft room acoustics
        nyquist = 0.5 * sr
        cutoff = min(1200.0, nyquist * 0.9)
        b, a = signal.butter(1, cutoff / nyquist, btype='low')
        filtered_tone = signal.lfilter(b, a, noise).astype(np.float32)

        return filtered_tone

    def equal_power_crossfade(
        self,
        audio_a: np.ndarray,
        audio_b: np.ndarray,
        sr: int
    ) -> np.ndarray:
        """
        Performs an equal-power crossfade between two audio segments.
        Equal-power curve: cos(t * pi / 2) and sin(t * pi / 2).
        """
        fade_samples = int((self.config.crossfade_ms / 1000.0) * sr)
        fade_samples = min(fade_samples, len(audio_a), len(audio_b))

        if fade_samples <= 0:
            return np.concatenate([audio_a, audio_b])

        # Time parameter normalized 0.0 -> 1.0
        t = np.linspace(0, 1, fade_samples, endpoint=True, dtype=np.float32)
        fade_out = np.cos(t * (np.pi / 2.0))
        fade_in = np.sin(t * (np.pi / 2.0))

        # Overlap region
        overlap_a = audio_a[-fade_samples:] * fade_out
        overlap_b = audio_b[:fade_samples] * fade_in
        blended = overlap_a + overlap_b

        # Concatenate: audio_a body + blended overlap + audio_b tail
        return np.concatenate([
            audio_a[:-fade_samples],
            blended,
            audio_b[fade_samples:]
        ])

    def apply_boundary_fades(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Applies gentle cosine fade-in and fade-out to prevent boundary clicks."""
        fade_len = int((self.config.fade_boundary_ms / 1000.0) * sr)
        if len(audio) < fade_len * 2 or fade_len <= 0:
            return audio

        out = audio.copy()
        t = np.linspace(0, 1, fade_len, endpoint=True, dtype=np.float32)
        fade_in = np.sin(t * (np.pi / 2.0))
        fade_out = np.cos(t * (np.pi / 2.0))

        out[:fade_len] *= fade_in
        out[-fade_len:] *= fade_out
        return out

    @timed_step("Audio Chunk Stitching")
    def stitch(
        self,
        chunks: List[np.ndarray],
        pauses_ms: List[int],
        sr: int = 24_000
    ) -> np.ndarray:
        """
        Stitches a sequence of audio chunks into a unified master audio array.

        Args:
            chunks: List of 1D float32 numpy arrays.
            pauses_ms: List of pause lengths (in ms) to follow each chunk.
            sr: Audio sample rate in Hz.

        Returns:
            1D float32 numpy array of seamless stitched audio.
        """
        if not chunks:
            return np.zeros(0, dtype=np.float32)

        if len(chunks) == 1:
            return self.apply_boundary_fades(chunks[0], sr)

        segments: List[np.ndarray] = []

        for i, chunk in enumerate(chunks):
            chunk_faded = self.apply_boundary_fades(chunk, sr)
            segments.append(chunk_faded)

            # Determine pause after this chunk
            if i < len(chunks) - 1:
                pause_duration_ms = pauses_ms[i] if i < len(pauses_ms) else 300
                pause_sec = max(0.05, pause_duration_ms / 1000.0)

                if self.config.insert_room_tone:
                    room_tone = self.generate_room_tone(pause_sec, sr)
                    # Crossfade room tone boundary
                    room_tone = self.apply_boundary_fades(room_tone, sr)
                    segments.append(room_tone)
                else:
                    silence = np.zeros(int(pause_sec * sr), dtype=np.float32)
                    segments.append(silence)

        # Concatenate all prepared segments
        stitched = np.concatenate(segments)
        logger.info(
            f"Stitched {len(chunks)} chunks into continuous audio "
            f"({len(stitched) / sr:.2f}s, sr={sr}Hz)"
        )
        return stitched.astype(np.float32)


audio_stitcher = AudioStitcher()
