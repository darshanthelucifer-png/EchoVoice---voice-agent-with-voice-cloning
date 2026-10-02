"""
ASR & Speech Streaming Unit Tests (backend/tests/test_asr.py)
-----------------------------------------------------------
PYTHON & ML CONCEPTS DEMONSTRATED:
- Abstract Base Class Verification: Asserts that FasterWhisperEngine and LocalFallbackASREngine
  strictly honor the ASREngine ABC contract.
- Registry Pattern & Graceful Degradation: Confirms dynamic resolution of ASR engines
  with automatic fallback on unknown identifiers.
- Streaming VAD-State Machine: Verifies event transitions (SILENCE -> SPEECH_STARTED -> INTERIM -> FINAL)
  across incoming chunked audio streams.
- WebSocket Integration Testing: Uses Starlette/FastAPI TestClient to validate bi-directional
  real-time WebSocket audio streaming protocols.
"""

import io
import json
import pytest
import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from app.main import app
from app.ai.asr.base import ASREngine, ASRResult, ASRToken
from app.ai.asr.faster_whisper_engine import FasterWhisperEngine
from app.ai.asr.fallback_engine import LocalFallbackASREngine
from app.ai.asr.registry import asr_registry, get_asr_engine
from app.ai.asr.stream import VADAudioStreamProcessor, StreamEventType, StreamConfig
from app.services.asr_service import asr_service


def _generate_synthetic_tone(duration_sec: float = 1.0, sr: int = 16000, freq: float = 440.0) -> np.ndarray:
    """Generate a synthetic test tone with harmonic modulation."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)
    # Fundamental + 1st harmonic + slight envelope to simulate vocalic energy
    audio = 0.5 * np.sin(2 * np.pi * freq * t) + 0.25 * np.sin(2 * np.pi * (freq * 2) * t)
    envelope = np.sin(np.pi * np.linspace(0, 1, len(audio), dtype=np.float32))
    return (audio * envelope).astype(np.float32)


def _audio_to_wav_bytes(audio: np.ndarray, sr: int = 16000) -> bytes:
    """Helper to convert float32 numpy array to standard WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ==============================================================================
# 1. Interface & Registry Tests
# ==============================================================================

def test_asr_engine_interface():
    """Verify that all registered engines adhere to the ASREngine ABC contract."""
    fallback = LocalFallbackASREngine()
    whisper = FasterWhisperEngine()

    for engine in [fallback, whisper]:
        assert isinstance(engine, ASREngine)
        assert isinstance(engine.engine_name, str)


def test_asr_registry_resolution():
    """Verify dynamic registry resolution and graceful fallback on unknown identifiers."""
    engine_whisper = get_asr_engine("faster_whisper")
    assert isinstance(engine_whisper, FasterWhisperEngine)

    engine_hyphen = get_asr_engine("faster-whisper")
    assert isinstance(engine_hyphen, FasterWhisperEngine)

    engine_fb = get_asr_engine("fallback")
    assert isinstance(engine_fb, LocalFallbackASREngine)

    # Unknown engine resolves gracefully to fallback engine without crashing
    unknown = get_asr_engine("non_existent_engine")
    assert isinstance(unknown, LocalFallbackASREngine)


# ==============================================================================
# 2. Engine Transcription Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_fallback_asr_transcribe():
    """Verify LocalFallbackASREngine generates valid ASRResult with token timings."""
    audio = _generate_synthetic_tone(duration_sec=1.5, sr=16000)
    engine = LocalFallbackASREngine()

    result = await engine.transcribe(audio, sr=16000, language="en", word_timestamps=True)

    assert isinstance(result, ASRResult)
    assert len(result.text) > 0
    assert result.duration_seconds == pytest.approx(1.5, rel=1e-2)
    assert result.rtf >= 0.0
    assert len(result.words) > 0
    assert isinstance(result.words[0], ASRToken)


@pytest.mark.asyncio
async def test_faster_whisper_transcribe():
    """Verify FasterWhisperEngine transcribes audio and measures latency & RTF."""
    audio = _generate_synthetic_tone(duration_sec=1.0, sr=16000)
    engine = FasterWhisperEngine()

    result = await engine.transcribe(audio, sr=16000, language="en", word_timestamps=True)

    assert isinstance(result, ASRResult)
    assert result.duration_seconds == pytest.approx(1.0, rel=1e-2)
    assert result.rtf >= 0.0
    assert "faster_whisper" in result.engine_name


# ==============================================================================
# 3. Streaming VAD Processor Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_vad_audio_stream_processor():
    """
    Verify VADAudioStreamProcessor state machine:
    1. Silence frames -> NO events
    2. Speech frames -> SPEECH_STARTED event, followed by INTERIM hypotheses
    3. Trailing silence -> FINAL utterance event with completed transcription
    """
    engine = LocalFallbackASREngine()
    config = StreamConfig(
        sample_rate=16000,
        silence_timeout_sec=0.20,
        min_speech_duration_sec=0.20,
        interim_interval_sec=0.30
    )
    processor = VADAudioStreamProcessor(engine=engine, config=config)

    # 1. Feed pure silence (amplitude 0.0) -> No speech event
    silence_chunk = np.zeros(1024, dtype=np.float32)
    for _ in range(5):
        events = await processor.process_chunk(silence_chunk)
        assert len(events) == 0

    # 2. Feed speech tone chunks (amplitude > 0.05 triggers VAD energy)
    speech_tone = _generate_synthetic_tone(duration_sec=0.064, sr=16000)  # 1024 samples
    started_seen = False

    for _ in range(10):  # ~640ms of active audio
        events = await processor.process_chunk(speech_tone)
        for ev in events:
            if ev["type"] == StreamEventType.SPEECH_STARTED.value:
                started_seen = True

    assert started_seen is True, "SPEECH_STARTED event must trigger on active speech"

    # 3. Feed silence to trigger utterance completion (silence_timeout_sec=0.20s is 3200 samples)
    final_seen = False
    for _ in range(10):  # 10240 samples of silence = 0.64s
        events = await processor.process_chunk(silence_chunk)
        for ev in events:
            if ev["type"] == StreamEventType.FINAL.value:
                final_seen = True
                assert "text" in ev
                assert ev["duration_seconds"] > 0.0

    assert final_seen is True, "FINAL event must trigger after trailing silence"

    # 4. Processor reset cleans internal buffers
    processor.reset()
    assert processor.is_speaking is False
    assert len(processor.speech_frames) == 0


# ==============================================================================
# 4. ASR Service Integration Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_asr_service_transcribe_audio_bytes():
    """Verify asr_service accurately decodes container bytes and executes transcription."""
    audio = _generate_synthetic_tone(duration_sec=1.0, sr=16000)
    wav_bytes = _audio_to_wav_bytes(audio, sr=16000)

    result = await asr_service.transcribe_audio_bytes(
        audio_bytes=wav_bytes,
        engine_name="fallback",
        language="en",
        word_timestamps=True
    )

    assert isinstance(result, ASRResult)
    assert len(result.text) > 0
    assert result.duration_seconds == pytest.approx(1.0, rel=1e-2)


# ==============================================================================
# 5. REST API Endpoint Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_api_asr_engines(client):
    """Verify GET /api/v1/asr/engines returns supported engine registry list."""
    response = await client.get("/api/v1/asr/engines")
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    engines = body["data"]["engines"]
    assert "faster_whisper" in engines
    assert "fallback" in engines


@pytest.mark.asyncio
async def test_api_asr_transcribe_upload(client):
    """Verify POST /api/v1/asr/transcribe accepts audio file uploads and returns transcript."""
    audio = _generate_synthetic_tone(duration_sec=1.0, sr=16000)
    wav_bytes = _audio_to_wav_bytes(audio, sr=16000)

    files = {"file": ("test_audio.wav", wav_bytes, "audio/wav")}
    data = {
        "engine": "fallback",
        "language": "en"
    }

    response = await client.post("/api/v1/asr/transcribe", files=files, data=data)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    res_data = body["data"]
    assert "text" in res_data
    assert "fallback" in res_data["engine_name"]
    assert res_data["duration_seconds"] == pytest.approx(1.0, rel=1e-1)
    assert len(res_data["words"]) > 0


# ==============================================================================
# 6. WebSocket Streaming Endpoint Tests
# ==============================================================================

def test_api_asr_websocket_streaming():
    """Verify WebSocket real-time audio streaming pipeline over /api/v1/asr/stream."""
    client = TestClient(app)

    with client.websocket_connect("/api/v1/asr/stream") as ws:
        # 1. Send configuration handshake
        ws.send_text(json.dumps({
            "action": "config",
            "engine": "fallback",
            "language": "en",
            "format": "int16"
        }))

        # 2. Receive READY acknowledgment
        msg = ws.receive_json()
        assert msg["type"] == "READY"
        assert msg["engine"] == "fallback"

        # 3. Send binary audio chunks simulating speech (5 * 0.1s = 0.5s > 0.3s min threshold)
        audio_chunk = _generate_synthetic_tone(duration_sec=0.1, sr=16000)
        pcm_bytes = (audio_chunk * 32767).astype(np.int16).tobytes()

        # First chunk triggers speech_started
        ws.send_bytes(pcm_bytes)
        msg_speech = ws.receive_json()
        assert msg_speech["type"] == "speech_started"

        # Subsequent speech chunks accumulate speech duration
        for _ in range(4):
            ws.send_bytes(pcm_bytes)

        # 4. Trigger flush to finalize utterance
        ws.send_text(json.dumps({"action": "flush"}))
        msg_final = ws.receive_json()
        assert msg_final["type"] == "final"
        assert "text" in msg_final
        assert msg_final["duration_seconds"] >= 0.5

        msg_ack = ws.receive_json()
        assert msg_ack["type"] == "flush_ack"

        # 5. Clean close
        ws.send_text(json.dumps({"action": "close"}))
