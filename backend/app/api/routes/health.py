"""
Health & Diagnostic Router (backend/app/api/routes/health.py)
--------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Health Checks & Probing: Tests underlying dependencies (DB engine connectivity)
  before declaring the service healthy.
- Information Hiding / Security: Safely returns configured model names without
  ever exposing secret keys or tokens.
"""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.api.deps import get_db
from app.core.config import settings
from app.schemas.common import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="System health check")
async def health_check(db: AsyncSession = Depends(get_db)) -> HealthResponse:
    """
    Returns system status, environment details, and verifies active DB connection.
    """
    db_connected = False
    try:
        result = await db.execute(text("SELECT 1"))
        db_connected = result.scalar() == 1
    except Exception:
        db_connected = False

    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        environment=settings.APP_ENV,
        database_connected=db_connected,
        configured_models={
            "asr_model": settings.ASR_MODEL,
            "llm_model": settings.LLM_MODEL,
            "embed_model": settings.EMBED_MODEL,
            "rerank_model": settings.RERANK_MODEL,
            "tts_engine": settings.TTS_ENGINE,
            "translate_model": settings.TRANSLATE_MODEL,
            "enhance_model": settings.ENHANCE_MODEL,
            "mastering_preset": settings.DEFAULT_MASTERING_PRESET,
        }
    )
