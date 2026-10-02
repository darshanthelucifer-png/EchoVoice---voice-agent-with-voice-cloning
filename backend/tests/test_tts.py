"""
TTS & Voice Cloning Unit Tests (backend/tests/test_tts.py)
----------------------------------------------------------
Verifies Phase 3 Text-to-Speech Engine Interface and A/B Testing:
1. TTSEngine interface compliance and capabilities
2. TTSRegistry dynamic resolution and fallback behavior
3. Short-text voice synthesis and speaker conditioning
4. Side-by-side A/B performance and perceptual benchmarking
5. HTTP endpoints (/tts/engines, /tts/synthesize, /tts/ab-test, /tts/audio/{filename})
"""

import pytest
import numpy as np

from app.ai.tts.base import TTSEngine, TTSOutput
from app.ai.tts.fallback_engine import LocalFallbackEngine
from app.ai.tts.xtts_engine import XTTSEngine
from app.ai.tts.registry import tts_registry, get_tts_engine
from app.services.tts_service import tts_service
from app.ai.audio.cleanup import AudioCleanupPipeline


def test_tts_engine_interface():
    """Verify engines adhere to TTSEngine Abstract Base Class."""
    fallback = LocalFallbackEngine()
    xtts = XTTSEngine()

    for engine in [fallback, xtts]:
        assert isinstance(engine, TTSEngine)
        assert len(engine.supported_languages()) > 0
        assert engine.is_available() is True
        assert engine.is_voice_cloning_supported() is True


def test_tts_registry_resolution():
    """Verify registry resolves engines and gracefully falls back on unknown keys."""
    engine_xtts = get_tts_engine("xtts_v2")
    assert isinstance(engine_xtts, XTTSEngine)

    engine_fb = get_tts_engine("fallback")
    assert isinstance(engine_fb, LocalFallbackEngine)

    # Unknown engine resolves to fallback
    unknown = get_tts_engine("non_existent_engine")
    assert isinstance(unknown, LocalFallbackEngine)


@pytest.mark.asyncio
async def test_short_text_synthesis():
    """Verify speech synthesis outputs valid audio array and persists to exports."""
    text = "EchoVoice phase three test."
    output, file_path = await tts_service.generate_speech(
        text=text,
        engine_name="fallback",
        language="en"
    )

    assert isinstance(output, TTSOutput)
    assert len(output.audio) > 0
    assert output.sample_rate == 24000
    assert output.duration_seconds > 0.3
    assert output.latency_ms > 0
    assert file_path.exists()


@pytest.mark.asyncio
async def test_voice_cloning_conditioning(tmp_path):
    """Verify synthesis conditioned on a reference clip captures reference metadata."""
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    ref_audio = (0.3 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    ref_path = tmp_path / "speaker_ref.wav"
    AudioCleanupPipeline.save_audio(ref_audio, sr, ref_path)

    output, _ = await tts_service.generate_speech(
        text="Conditioned speech test.",
        speaker_wav=ref_path,
        engine_name="fallback"
    )

    assert output.speaker_reference == str(ref_path)
    assert output.duration_seconds > 0.3


@pytest.mark.asyncio
async def test_ab_testing_service():
    """Verify side-by-side A/B synthesis comparison metrics."""
    result = await tts_service.run_ab_test(
        text="A/B perceptual testing comparison.",
        engine_name_a="fallback",
        engine_name_b="fallback"
    )

    assert "option_a" in result
    assert "option_b" in result
    assert result["option_a"]["duration_seconds"] > 0
    assert result["option_b"]["duration_seconds"] > 0
    assert "latency_delta_ms" in result


@pytest.mark.asyncio
async def test_tts_api_endpoints(client):
    """Verify REST API endpoints for TTS listing, synthesis, A/B testing, and download."""
    # 1. GET /api/v1/tts/engines
    res_engines = await client.get("/api/v1/tts/engines")
    assert res_engines.status_code == 200
    data_eng = res_engines.json()
    assert "xtts_v2" in data_eng["data"]["registered_engines"]

    # 2. POST /api/v1/tts/synthesize
    res_synth = await client.post("/api/v1/tts/synthesize", json={
        "text": "API endpoint synthesis verification.",
        "language": "en",
        "engine": "fallback"
    })
    assert res_synth.status_code == 200
    data_synth = res_synth.json()
    assert data_synth["success"] is True
    assert "download_url" in data_synth["data"]

    # 3. GET audio download
    download_url = data_synth["data"]["download_url"]
    res_audio = await client.get(download_url)
    assert res_audio.status_code == 200
    assert len(res_audio.content) > 100

    # 4. POST /api/v1/tts/ab-test
    res_ab = await client.post("/api/v1/tts/ab-test", json={
        "text": "A/B comparison endpoint test.",
        "engine_a": "fallback",
        "engine_b": "fallback",
        "language": "en"
    })
    assert res_ab.status_code == 200
    data_ab = res_ab.json()
    assert data_ab["success"] is True
    assert "option_a" in data_ab["data"]
    assert "option_b" in data_ab["data"]
