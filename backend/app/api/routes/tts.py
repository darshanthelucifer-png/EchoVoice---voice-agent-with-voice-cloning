"""
TTS & Voice Cloning Router (backend/app/api/routes/tts.py)
----------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- RESTful Audio Synthesis & Streaming Endpoints: Translates JSON requests into
  audio files or streams.
- A/B Perceptual Benchmark Endpoint: Exposes side-by-side engine and voice comparison
  for user evaluation.
- OpenAPI Tags & Documentation: Detailed parameter schemas for Swagger UI.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.schemas.common import APIResponse
from app.services.tts_service import tts_service
from app.ai.tts.registry import tts_registry

router = APIRouter(prefix="/tts", tags=["TTS & Voice Cloning"])


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Short text to synthesize")
    language: str = Field(default="en", description="Target ISO language code (e.g. 'en', 'es', 'hi')")
    speaker_wav_path: Optional[str] = Field(None, description="Path to clean reference audio for voice cloning")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed factor")
    emotion: Optional[str] = Field(None, description="Emotion style hint ('calm', 'warm', 'excited', 'serious')")
    engine: Optional[str] = Field(None, description="Override default TTS engine ('xtts_v2', 'fallback')")


class ABTestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Text script to compare across options")
    engine_a: str = Field(default="xtts_v2", description="Engine for Option A")
    engine_b: str = Field(default="fallback", description="Engine for Option B")
    speaker_wav_path: Optional[str] = Field(None, description="Reference audio clip to clone")
    language: str = Field(default="en", description="Target language")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@router.get("/engines", summary="List registered TTS engines and capabilities")
async def list_engines():
    """Returns all registered TTS engines in the system."""
    engines = tts_registry.available_engines()
    return APIResponse(
        success=True,
        data={
            "active_engine": settings.TTS_ENGINE,
            "registered_engines": engines,
        }
    )


@router.post("/synthesize", summary="Synthesize short text with voice cloning (Phase 3)")
async def synthesize_text(req: SynthesisRequest):
    """
    Synthesizes short text into spoken audio conditioned on the speaker reference clip.
    """
    try:
        output, save_path = await tts_service.generate_speech(
            text=req.text,
            speaker_wav=req.speaker_wav_path,
            language=req.language,
            speed=req.speed,
            emotion=req.emotion,
            engine_name=req.engine
        )

        return APIResponse(
            success=True,
            message="Speech synthesized successfully",
            data={
                "download_url": f"/api/v1/tts/audio/{save_path.name}",
                "filename": save_path.name,
                "duration_seconds": output.duration_seconds,
                "latency_ms": output.latency_ms,
                "rtf": round(output.rtf, 3),
                "engine": output.engine_name,
                "language": output.language,
                "metadata": output.metadata
            }
        )
    except Exception as exc:
        logger.error(f"Synthesis failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis error: {str(exc)}"
        )


@router.post("/ab-test", summary="Run side-by-side A/B comparison test (Phase 3)")
async def run_ab_test_endpoint(req: ABTestRequest):
    """
    Synthesizes the exact same text across two engines or reference voices
    to evaluate timbre and latency.
    """
    try:
        results = await tts_service.run_ab_test(
            text=req.text,
            speaker_wav_a=req.speaker_wav_path,
            engine_name_a=req.engine_a,
            engine_name_b=req.engine_b,
            language=req.language,
            speed=req.speed
        )
        return APIResponse(
            success=True,
            message="A/B test completed successfully",
            data=results
        )
    except Exception as exc:
        logger.error(f"A/B test failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test execution error: {str(exc)}"
        )


@router.get("/audio/{filename}", summary="Download or stream synthesized speech")
async def get_synthesized_audio(filename: str):
    """Streams the synthesized WAV file for playback in the web player."""
    file_path = settings.EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{filename}' not found."
        )
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )
