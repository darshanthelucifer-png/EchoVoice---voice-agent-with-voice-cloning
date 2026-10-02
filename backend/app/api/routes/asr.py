"""
Speech-to-Text & Real-Time ASR Router (backend/app/api/routes/asr.py)
---------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Full-Duplex WebSockets (`WebSocket`): Asynchronously ingests binary PCM audio frames
  and emits JSON events (`speech_started`, `interim`, `final`) with sub-200ms latency.
- Multipart File Upload Handling: Ingests audio files in any container format (WAV, MP3, WebM, FLAC).
"""

import json
from typing import Optional
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
import numpy as np

from app.core.logging import logger
from app.schemas.common import APIResponse
from app.schemas.asr import ASRResponse, ASRTokenSchema
from app.services.asr_service import asr_service
from app.ai.asr.registry import asr_registry

router = APIRouter(prefix="/asr", tags=["Speech-to-Text (ASR)"])


@router.get("/engines", summary="List registered ASR speech-to-text engines")
async def list_asr_engines():
    """Returns available ASR engines in the system."""
    return APIResponse(
        success=True,
        data={
            "engines": asr_registry.available_engines(),
            "default_engine": "faster_whisper"
        }
    )


@router.post(
    "/transcribe",
    response_model=APIResponse[ASRResponse],
    summary="Transcribe audio file or clip"
)
async def transcribe_audio_file(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, WebM, FLAC)"),
    language: Optional[str] = Form(None, description="Optional ISO language code (e.g. 'en', 'es', 'hi')"),
    engine: Optional[str] = Form(None, description="Optional engine override ('faster_whisper', 'fallback')")
):
    """
    Transcribes an uploaded audio file into text with word-level timestamps.
    """
    audio_bytes = await file.read()
    if len(audio_bytes) < 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty or corrupted."
        )

    try:
        result = await asr_service.transcribe_audio_bytes(
            audio_bytes=audio_bytes,
            language=language,
            engine_name=engine,
            word_timestamps=True
        )

        tokens = [
            ASRTokenSchema(
                word=w.word,
                start_sec=w.start_sec,
                end_sec=w.end_sec,
                probability=w.probability
            )
            for w in result.words
        ]

        response_payload = ASRResponse(
            text=result.text,
            language=result.language,
            duration_seconds=result.duration_seconds,
            latency_ms=result.latency_ms,
            rtf=result.rtf,
            confidence=result.confidence,
            engine_name=result.engine_name,
            words=tokens
        )

        return APIResponse(
            success=True,
            message="Audio transcribed successfully.",
            data=response_payload
        )
    except Exception as exc:
        logger.error(f"Transcription error: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech recognition error: {str(exc)}"
        )


@router.websocket("/stream")
async def websocket_asr_stream(websocket: WebSocket):
    """
    Real-time streaming ASR WebSocket endpoint.
    Client sends binary audio chunks (Float32 or Int16 at 16,000 Hz) or JSON control commands.
    Server evaluates Silero VAD and emits:
      - {"type": "speech_started", "timestamp": ...}
      - {"type": "interim", "text": "...", "duration": ...}
      - {"type": "final", "text": "...", "latency_ms": ..., "rtf": ..., "words": [...]}
    """
    await websocket.accept()
    processor = asr_service.create_stream_processor()
    stream_format = "auto"
    logger.info("Real-time ASR WebSocket streaming session established.")

    try:
        while True:
            message = await websocket.receive()

            # Handle JSON control frames
            if "text" in message and message["text"]:
                try:
                    payload = json.loads(message["text"])
                    action = payload.get("action") or payload.get("type")
                    if "format" in payload:
                        stream_format = payload["format"]

                    if action in ("flush", "close"):
                        events = await processor.flush()
                        for ev in events:
                            await websocket.send_json(ev)
                        if action == "flush":
                            await websocket.send_json({"type": "flush_ack", "events_emitted": len(events)})
                        elif action == "close":
                            break
                    elif action == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif action == "reset":
                        processor = asr_service.create_stream_processor()
                        await websocket.send_json({"type": "reset_ack"})
                    elif action == "config":
                        engine_name = payload.get("engine")
                        processor = asr_service.create_stream_processor(engine_name=engine_name)
                        await websocket.send_json({"type": "READY", "engine": engine_name or "faster_whisper"})
                except Exception as exc:
                    logger.debug(f"JSON frame parse error: {exc}")

            # Handle Binary PCM Audio Frames
            elif "bytes" in message and message["bytes"]:
                raw_bytes = message["bytes"]
                if stream_format == "int16":
                    int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
                    chunk = (int16_data / 32768.0).astype(np.float32)
                elif stream_format == "float32":
                    chunk = np.frombuffer(raw_bytes, dtype=np.float32).copy()
                else:
                    # Auto-detect: if float32 yields plausible normalised values in [-1.5, 1.5]
                    if len(raw_bytes) % 4 == 0:
                        f_candidate = np.frombuffer(raw_bytes, dtype=np.float32)
                        if len(f_candidate) > 0 and np.max(np.abs(f_candidate)) <= 1.5 and not np.all(f_candidate == 0):
                            chunk = f_candidate.copy()
                        else:
                            int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
                            chunk = (int16_data / 32768.0).astype(np.float32)
                    elif len(raw_bytes) % 2 == 0:
                        int16_data = np.frombuffer(raw_bytes, dtype=np.int16)
                        chunk = (int16_data / 32768.0).astype(np.float32)
                    else:
                        continue

                events = await processor.process_chunk(chunk)
                for ev in events:
                    await websocket.send_json(ev)

    except WebSocketDisconnect:
        logger.info("ASR WebSocket streaming disconnected cleanly.")
    except Exception as exc:
        logger.warning(f"ASR WebSocket exception: {exc}")
