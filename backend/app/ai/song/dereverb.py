"""
Song Studio Vocal Dereverberation & Bleed Removal (backend/app/ai/song/dereverb.py)
----------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Transient-Aware Spectral Decay Gating: Distinguishes direct vocal onsets from diffuse late
  reverberant energy by computing multi-band spectral flux and applying envelope-decay attenuation.
- High-Pass Rumble & Spill Rejection: Steep Butterworth 4th-order filter eliminating sub-bass
  kick/bass bleed below 85 Hz that leaks into the vocal stem.
- Adaptive Stationary Bleed Suppression: Tracks low-magnitude inter-phrase floor noise to eliminate
  headphone click bleed and background synth spill.
- Phase-Preserving STFT Reconstruction: Uses inverse STFT with Hann window and Griffin-Lim /
  original phase preservation to avoid comb filtering and robotic artifacts.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, Dict, Any
import time
import asyncio
import numpy as np
import librosa
from scipy import signal
import soundfile as sf

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.audio.cleanup import AudioCleanupPipeline


@dataclass
class DereverbConfig:
    """Parameters governing vocal dereverberation and bleed cancellation."""
    strength: float = 0.50                 # 0.0 (bypass) to 1.0 (aggressive de-reverb)
    highpass_cutoff_hz: float = 85.0       # Rejection filter for sub-bass bleed
    de_bleed_threshold_db: float = -42.0   # Spectral floor for residual spill suppression
    decay_suppression_factor: float = 0.65 # Attenuation ratio on late reverb tails
    n_fft: int = 2048
    hop_length: int = 512


@dataclass
class DereverbResult:
    """Processed dereverberated vocal audio payload."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    reverb_attenuation_db: float           # Estimated reduction of reverberant tail energy
    latency_ms: float
    output_path: Optional[Path] = None


class VocalDereverberator:
    """
    Strips room reflections, plate reverb tails, and headphone/instrumental bleed
    from isolated vocal tracks prior to voice conversion.
    """

    def __init__(self, config: Optional[DereverbConfig] = None):
        self.config = config or DereverbConfig(
            strength=settings.VOCAL_DEREVERB_STRENGTH
        )

    def process(self, audio: np.ndarray, sr: int, config: Optional[DereverbConfig] = None) -> DereverbResult:
        """
        Synchronously applies multi-band transient-aware dereverberation to vocal audio.
        """
        t0 = time.perf_counter()
        cfg = config or self.config

        if len(audio) == 0:
            return DereverbResult(
                audio=audio,
                sample_rate=sr,
                duration_seconds=0.0,
                reverb_attenuation_db=0.0,
                latency_ms=0.0
            )

        # Ensure 1D float32
        sig = audio.astype(np.float32)
        if sig.ndim > 1:
            sig = np.mean(sig, axis=0)

        # 1. High-Pass Filter: Eliminate sub-bass rumble and kick drum bleed
        sos = signal.butter(4, cfg.highpass_cutoff_hz, btype="highpass", fs=sr, output="sos")
        hp_sig = signal.sosfilt(sos, sig)

        # If strength is 0, return high-passed signal
        if cfg.strength <= 0.01:
            latency = (time.perf_counter() - t0) * 1000.0
            return DereverbResult(
                audio=hp_sig,
                sample_rate=sr,
                duration_seconds=round(len(hp_sig) / sr, 2),
                reverb_attenuation_db=0.0,
                latency_ms=round(latency, 2)
            )

        # 2. STFT Spectral Analysis
        stft = librosa.stft(hp_sig, n_fft=cfg.n_fft, hop_length=cfg.hop_length, window="hann")
        mag, phase = np.abs(stft), np.angle(stft)

        # 3. Spectral Flux Onset Detection: Identify dry vocal attacks
        # Spectral flux = positive difference between consecutive frames
        flux = np.diff(mag, axis=1, prepend=mag[:, :1])
        flux_pos = np.maximum(0.0, flux)

        # Normalize flux per frequency bin
        flux_norm = flux_pos / (np.max(flux_pos, axis=1, keepdims=True) + 1e-6)

        # 4. Compute Late Reverb Tail Gain Mask
        # Where flux is high -> dry transient -> keep gain near 1.0
        # Where energy decays slowly with low flux -> diffuse late reverb -> attenuate
        onset_mask = np.clip(flux_norm * 3.0, 0.0, 1.0)

        # Rolling exponential envelope to model reverb decay
        decay_mask = np.zeros_like(mag)
        alpha = 0.82  # Decay smoothing memory
        accum = mag[:, 0]
        decay_mask[:, 0] = accum

        for t_idx in range(1, mag.shape[1]):
            # If new onset, reset accumulation; otherwise decay
            accum = np.maximum(mag[:, t_idx], accum * alpha)
            decay_mask[:, t_idx] = accum

        # Contrast ratio between instantaneous energy and sustained decay envelope
        ratio = mag / (decay_mask + 1e-6)
        # Low ratio indicates diffuse tail energy; high ratio indicates direct sound
        tail_attenuation = np.clip(ratio ** (cfg.strength * 1.5), 0.25, 1.0)

        # Combine onset preservation with tail suppression
        gain_mask = (1.0 - cfg.strength) + cfg.strength * (0.4 * onset_mask + 0.6 * tail_attenuation)
        gain_mask = np.clip(gain_mask, 0.15, 1.0)

        # 5. Residual De-Bleed: Suppress low-level floor bleed
        mag_db = librosa.amplitude_to_db(mag, ref=np.max)
        bleed_mask = np.where(mag_db < cfg.de_bleed_threshold_db, 0.35, 1.0)

        # Apply combined gains to magnitude
        clean_mag = mag * gain_mask * bleed_mask

        # Estimate reverb tail attenuation in dB
        orig_energy = np.sum(mag ** 2) + 1e-9
        clean_energy = np.sum(clean_mag ** 2) + 1e-9
        attenuation_db = round(float(10.0 * np.log10(orig_energy / clean_energy)), 2)

        # 6. Inverse STFT Phase Reconstruction
        clean_stft = clean_mag * np.exp(1j * phase)
        clean_audio = librosa.istft(clean_stft, hop_length=cfg.hop_length, window="hann", length=len(hp_sig))

        # Peak normalization to prevent clipping while maintaining dynamic integrity
        peak = np.max(np.abs(clean_audio)) + 1e-7
        if peak > 0.95:
            clean_audio = clean_audio * (0.95 / peak)

        latency = (time.perf_counter() - t0) * 1000.0

        return DereverbResult(
            audio=clean_audio.astype(np.float32),
            sample_rate=sr,
            duration_seconds=round(len(clean_audio) / sr, 2),
            reverb_attenuation_db=attenuation_db,
            latency_ms=round(latency, 2)
        )

    @timed_step("Vocal Dereverberation & Bleed Removal")
    async def process_async(
        self,
        audio_input: Union[str, Path, np.ndarray],
        sr: int = 44_100,
        output_path: Optional[Path] = None,
        config: Optional[DereverbConfig] = None
    ) -> DereverbResult:
        """Asynchronously cleans and dereverberates vocal audio."""
        if isinstance(audio_input, (str, Path)):
            audio, sample_rate = AudioCleanupPipeline.load_audio(Path(audio_input), target_sr=sr)
        else:
            audio = audio_input
            sample_rate = sr

        result = await asyncio.to_thread(self.process, audio, sample_rate, config)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(AudioCleanupPipeline.save_audio, result.audio, result.sample_rate, output_path)
            result.output_path = output_path

        return result


# Global singleton instance
vocal_dereverberator = VocalDereverberator()
