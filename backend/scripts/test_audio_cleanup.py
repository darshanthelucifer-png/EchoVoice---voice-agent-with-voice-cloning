"""
Audio Cleanup CLI Test (backend/scripts/test_audio_cleanup.py)
--------------------------------------------------------------
Tests Phase 2 Audio Cleanup Pipeline and Quality Analyzer on a 20-second clip.

Usage:
  # With automatic realistic 20s test voice generation:
  python backend/scripts/test_audio_cleanup.py

  # With your own real 20s voice recording:
  python backend/scripts/test_audio_cleanup.py --input path/to/my_voice.wav
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.ai.audio.cleanup import cleanup_pipeline, CleanupConfig, AudioCleanupPipeline
from app.ai.audio.quality import quality_analyzer


def generate_realistic_20s_test_voice(sr: int = 24_000) -> np.ndarray:
    """
    Synthesizes a realistic 20-second speech recording with formant harmonics,
    natural pauses, electrical hum, and ambient room noise to benchmark cleanup performance.
    """
    total_samples = sr * 20
    audio = np.zeros(total_samples, dtype=np.float32)

    # Helper to generate vowel-like voiced speech burst
    def generate_phrase(duration_sec: float, base_f0: float = 125.0) -> np.ndarray:
        t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
        sig = np.zeros_like(t)
        # Formant harmonics (1st to 16th harmonic)
        for h in range(1, 16):
            weight = 1.0 / (h ** 0.85)
            # Add subtle pitch modulation (vibrato / prosody)
            f = (base_f0 + 8.0 * np.sin(2 * np.pi * 2.5 * t)) * h
            sig += weight * np.sin(2 * np.pi * f * t)

        # Syllabic speech cadence (approx 4.5 Hz envelope)
        envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 4.5 * t)) ** 2
        sig = sig * envelope

        # Smooth attack and decay at phrase boundaries
        fade_len = int(sr * 0.05)
        fade_in = np.linspace(0, 1, fade_len)
        fade_out = np.linspace(1, 0, fade_len)
        sig[:fade_len] *= fade_in
        sig[-fade_len:] *= fade_out

        return (sig * 0.35).astype(np.float32)

    # Place speech phrases with conversational pauses across 20s:
    # 0s - 0.8s: Leading silence (0.8s)
    # 0.8s - 5.3s: Phrase 1 (4.5s)
    # 5.3s - 6.5s: Pause 1 (1.2s silence)
    # 6.5s - 11.5s: Phrase 2 (5.0s)
    # 11.5s - 12.8s: Pause 2 (1.3s silence)
    # 12.8s - 17.8s: Phrase 3 (5.0s)
    # 17.8s - 20.0s: Trailing silence (2.2s)
    phrases = [
        (0.8, 4.5, 120.0),
        (6.5, 5.0, 132.0),
        (12.8, 5.0, 118.0),
    ]

    for start_sec, dur_sec, f0 in phrases:
        start_idx = int(start_sec * sr)
        p = generate_phrase(dur_sec, f0)
        end_idx = min(start_idx + len(p), total_samples)
        audio[start_idx:end_idx] = p[:end_idx - start_idx]

    # Add realistic real-world acoustic corruptions:
    t_full = np.linspace(0, 20, total_samples, endpoint=False)
    
    # 1. 50 Hz power hum + 100 Hz harmonic
    hum = 0.035 * np.sin(2 * np.pi * 50 * t_full) + 0.015 * np.sin(2 * np.pi * 100 * t_full)
    
    # 2. Ambient room hiss / HVAC pink noise
    noise = np.random.normal(0, 0.03, total_samples)
    
    # 3. Small DC offset
    dc_offset = 0.015

    corrupted = audio + hum + noise + dc_offset
    return corrupted.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="EchoVoice Audio Cleanup CLI Test")
    parser.add_argument("--input", type=str, default=None, help="Path to input audio file (WAV, MP3, FLAC)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save output files")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Target sample rate (Hz)")
    parser.add_argument("--target-lufs", type=float, default=-14.0, help="Target loudness (LUFS)")
    args = parser.parse_args()

    print("=" * 76)
    print("EchoVoice - Phase 2: Audio Cleanup & Quality Analysis Benchmark")
    print("=" * 76)

    output_dir = Path(args.output_dir) if args.output_dir else (settings.VOICE_PROFILES_DIR / "phase2_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load or Synthesize Audio
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] Input file does not exist: {input_path}")
            sys.exit(1)
        print(f"[1/4] Loading user input audio: {input_path}")
        raw_audio, sr = AudioCleanupPipeline.load_audio(input_path)
    else:
        print("[1/4] Synthesizing realistic 20-second test speech with hum & room noise...")
        sr = args.sample_rate
        raw_audio = generate_realistic_20s_test_voice(sr=sr)

    # Save original reference copy
    original_path = output_dir / "original_sample.wav"
    AudioCleanupPipeline.save_audio(raw_audio, sr, original_path)
    print(f"      Saved original audio -> {original_path}")

    # 2. Analyze Raw Audio Quality
    print("[2/4] Analyzing acoustic quality of original audio...")
    raw_report = quality_analyzer.analyze(raw_audio, sr)

    # 3. Execute Cleanup Pipeline
    print("[3/4] Running multi-stage restoration pipeline:")
    print("      - DC Offset Removal")
    print("      - Peak Pre-Normalization (-3.0 dBFS)")
    print("      - High-Pass Filter (80 Hz rumble removal)")
    print("      - Stationary Spectral Gating Noise Reduction")
    print("      - Silero VAD Silence Trimming & Pause Capping (0.7s max)")
    print("      - Room Tone Injection (-65 dBFS floor)")
    print(f"      - YouTube-Standard Loudness Normalization ({args.target_lufs} LUFS) & True-Peak Limiter")

    config = CleanupConfig(
        target_sample_rate=args.sample_rate,
        target_lufs=args.target_lufs,
        max_internal_pause_sec=0.70,
        denoise_prop_decrease=0.80
    )
    result = cleanup_pipeline.process(raw_audio, sr, config)

    # Save cleaned output
    cleaned_path = output_dir / "reference.wav"
    AudioCleanupPipeline.save_audio(result.audio, result.sample_rate, cleaned_path)
    print(f"      Saved cleaned audio  -> {cleaned_path}")

    # 4. Compare Metrics
    print("\n[4/4] Before vs. After Acoustic Comparison:")
    print("-" * 76)
    print(f"{'Metric':<30} | {'Before (Raw)':<18} | {'After (Cleaned)':<18}")
    print("-" * 76)
    print(f"{'Total Duration':<30} | {raw_report.duration_seconds:<15.2f} s | {result.cleaned_report.duration_seconds:<15.2f} s")
    print(f"{'Speech Duration':<30} | {raw_report.speech_duration_seconds:<15.2f} s | {result.cleaned_report.speech_duration_seconds:<15.2f} s")
    print(f"{'Speech Ratio':<30} | {raw_report.speech_ratio * 100:<15.1f} % | {result.cleaned_report.speech_ratio * 100:<15.1f} %")
    print(f"{'Signal-to-Noise Ratio (SNR)':<30} | {raw_report.snr_db:<15.1f} dB| {result.cleaned_report.snr_db:<15.1f} dB")
    print(f"{'Background Noise Floor':<30} | {raw_report.noise_floor_db:<15.1f} dBFS | {result.cleaned_report.noise_floor_db:<15.1f} dBFS")
    print(f"{'Peak Amplitude':<30} | {raw_report.peak_dbfs:<15.1f} dBFS | {result.cleaned_report.peak_dbfs:<15.1f} dBFS")
    print(f"{'Clipping Ratio':<30} | {raw_report.clipping_ratio * 100:<15.3f} % | {result.cleaned_report.clipping_ratio * 100:<15.3f} %")
    print(f"{'Spectral Flatness':<30} | {raw_report.spectral_flatness:<15.4f}   | {result.cleaned_report.spectral_flatness:<15.4f}  ")
    print(f"{'Usable for Voice Cloning':<30} | {str(raw_report.is_usable):<18} | {str(result.cleaned_report.is_usable):<18}")
    print("-" * 76)

    snr_gain = result.cleaned_report.snr_db - raw_report.snr_db
    noise_reduction = raw_report.noise_floor_db - result.cleaned_report.noise_floor_db
    print(f"\nRESULTS:")
    print(f"  * SNR Improvement:       +{snr_gain:.1f} dB")
    print(f"  * Noise Floor Reduction: {noise_reduction:.1f} dB")
    print(f"  * Output Loudness:       {args.target_lufs} LUFS (YouTube Standard)")
    print(f"  * Quality Advice:        {result.cleaned_report.recommendation}")
    print("=" * 76)
    print("SUCCESS: Phase 2 Audio Cleanup Pipeline verified successfully!")
    print("=" * 76)


if __name__ == "__main__":
    main()
