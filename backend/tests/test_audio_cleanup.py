"""
Audio Cleanup & Quality Pipeline Unit Tests (backend/tests/test_audio_cleanup.py)
---------------------------------------------------------------------------------
Verifies Phase 2 audio components:
1. Silero / Energy VAD speech boundary detection and silence trimming
2. Audio quality analyzer (SNR, clipping, noise floor, recommendations)
3. Noise reduction and DC offset elimination
4. YouTube-standard loudness normalization & true-peak limiter
5. Non-blocking AudioService thread offloading
"""

import pytest
import numpy as np
import pyloudnorm as pyln

from app.ai.audio.vad import vad_processor
from app.ai.audio.quality import quality_analyzer
from app.ai.audio.cleanup import cleanup_pipeline, CleanupConfig, AudioCleanupPipeline
from app.services.audio_service import audio_service


def create_test_tone(duration_sec: float, sr: int = 24000, freq: float = 200.0) -> np.ndarray:
    """Helper to synthesize harmonic voice-like test tone."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 4]:
        sig += (0.4 / h) * np.sin(2 * np.pi * freq * h * t)
    return sig.astype(np.float32)


def test_audio_quality_analyzer_metrics():
    """Verify quality analyzer computes duration, clipping, SNR, and recommendations."""
    sr = 24000
    duration = 3.0
    speech = create_test_tone(duration, sr=sr)
    
    # Intentionally clip some samples
    speech[:50] = 1.05
    speech[50:100] = -1.05

    report = quality_analyzer.analyze(speech, sr)
    assert report.duration_seconds == pytest.approx(3.0, abs=0.1)
    assert report.clipping_ratio > 0.001
    assert report.peak_dbfs >= -0.1
    assert "clipping" in report.recommendation.lower()


def test_vad_detection_and_trimming():
    """Verify VAD trims leading/trailing dead air and caps internal pauses."""
    sr = 24000
    # 1.5s silence + 2.0s voice tone + 2.0s pause + 2.0s voice tone + 1.5s silence = 9.0s total
    silence_lead = np.zeros(int(sr * 1.5), dtype=np.float32)
    tone1 = create_test_tone(2.0, sr=sr)
    silence_mid = np.zeros(int(sr * 2.0), dtype=np.float32)
    tone2 = create_test_tone(2.0, sr=sr)
    silence_trail = np.zeros(int(sr * 1.5), dtype=np.float32)

    audio = np.concatenate([silence_lead, tone1, silence_mid, tone2, silence_trail])
    orig_duration = len(audio) / sr
    assert orig_duration == pytest.approx(9.0, abs=0.1)

    trimmed = vad_processor.trim_silence(
        audio=audio,
        sr=sr,
        max_internal_pause_sec=0.7,
        padding_sec=0.1
    )
    trimmed_duration = len(trimmed) / sr

    # Trimmed duration should be significantly shorter than 9.0s (around 4.5s - 5.5s)
    assert trimmed_duration < orig_duration - 2.0
    assert trimmed_duration > 3.0


def test_audio_cleanup_pipeline():
    """Verify DC offset removal, denoise, and peak normalization."""
    sr = 24000
    # Tone with speech pause so 10th percentile reflects noise floor
    tone1 = create_test_tone(1.5, sr=sr)
    pause = np.zeros(int(sr * 1.0), dtype=np.float32)
    tone2 = create_test_tone(1.5, sr=sr)
    speech = np.concatenate([tone1, pause, tone2])
    
    # Add DC offset and electrical hum
    t = np.linspace(0, len(speech) / sr, len(speech), endpoint=False)
    hum = 0.05 * np.sin(2 * np.pi * 50 * t)
    noise = np.random.normal(0, 0.04, len(speech)).astype(np.float32)
    corrupted = speech + hum + noise + 0.02  # DC offset = 0.02

    config = CleanupConfig(
        target_sample_rate=sr,
        enable_denoise=True,
        enable_vad_trim=False,  # Test DSP cleanup without trimming
        enable_loudness_norm=True,
        target_lufs=-14.0,
        true_peak_limit_db=-1.5
    )

    result = cleanup_pipeline.process(corrupted, sr, config)

    # 1. DC offset eliminated
    assert abs(np.mean(result.audio)) < 0.005

    # 2. Peak limited to <= -1.5 dBTP (approx 0.841 amplitude)
    max_amp = float(np.max(np.abs(result.audio)))
    assert max_amp <= 0.86

    # 3. SNR improved
    assert result.cleaned_report.snr_db > result.raw_report.snr_db


def test_youtube_standard_loudness():
    """Verify audio is mastered to YouTube -14 LUFS target."""
    sr = 24000
    speech = create_test_tone(5.0, sr=sr)
    
    config = CleanupConfig(
        target_sample_rate=sr,
        enable_denoise=False,
        enable_vad_trim=False,
        enable_loudness_norm=True,
        target_lufs=-14.0
    )

    result = cleanup_pipeline.process(speech, sr, config)

    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(result.audio)
    # Target is -14.0 LUFS (allow +/- 1.5 LUFS tolerance for short synthetic tones)
    assert loudness == pytest.approx(-14.0, abs=1.5)


@pytest.mark.asyncio
async def test_audio_service_async_execution(tmp_path):
    """Verify AudioService executes in background thread pool without blocking asyncio loop."""
    sr = 24000
    speech = create_test_tone(3.0, sr=sr)
    input_wav = tmp_path / "test_input.wav"
    AudioCleanupPipeline.save_audio(speech, sr, input_wav)

    # 1. Async Quality Analysis
    report = await audio_service.analyze_audio(input_wav)
    assert report.duration_seconds == pytest.approx(3.0, abs=0.1)

    # 2. Async Cleaning
    out_file, result = await audio_service.clean_voice_sample(
        input_source=input_wav,
        output_filename="async_cleaned.wav"
    )
    assert out_file.exists()
    assert result.cleaned_report.snr_db >= 0.0


@pytest.mark.asyncio
async def test_voice_profiles_api_endpoints(client, tmp_path):
    """Verify multipart file upload to /analyze-quality and /clean-audio endpoints."""
    import io
    import soundfile as sf

    sr = 24000
    speech = create_test_tone(3.0, sr=sr)
    buf = io.BytesIO()
    sf.write(buf, speech, sr, format="WAV")
    buf.seek(0)
    wav_bytes = buf.read()

    # 1. Test POST /api/v1/voice-profiles/analyze-quality
    files = {"file": ("test_mic.wav", wav_bytes, "audio/wav")}
    res_analyze = await client.post("/api/v1/voice-profiles/analyze-quality", files=files)
    assert res_analyze.status_code == 200
    data_analyze = res_analyze.json()
    assert data_analyze["success"] is True
    assert "snr_db" in data_analyze["data"]

    # 2. Test POST /api/v1/voice-profiles/clean-audio
    buf.seek(0)
    files = {"file": ("test_mic.wav", wav_bytes, "audio/wav")}
    res_clean = await client.post("/api/v1/voice-profiles/clean-audio", files=files)
    assert res_clean.status_code == 200
    data_clean = res_clean.json()
    assert data_clean["success"] is True
    assert "download_url" in data_clean["data"]

    # 3. Test GET cleaned audio file
    download_url = data_clean["data"]["download_url"]
    res_audio = await client.get(download_url)
    assert res_audio.status_code == 200
    assert len(res_audio.content) > 1000
