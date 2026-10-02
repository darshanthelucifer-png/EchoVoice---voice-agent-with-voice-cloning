"""
API Routes Package (backend/app/api/routes/__init__.py)
-------------------------------------------------------
Exports all individual API routers.
"""

from app.api.routes.auth import router as auth_router
from app.api.routes.health import router as health_router
from app.api.routes.voice_profiles import router as voice_profiles_router
from app.api.routes.tts import router as tts_router
from app.api.routes.tts_jobs import router as tts_jobs_router
from app.api.routes.asr import router as asr_router
from app.api.routes.rag import router as rag_router
from app.api.routes.knowledge import router as knowledge_router
from app.api.routes.chat import router as chat_router
from app.api.routes.metrics import router as metrics_router
from app.api.routes.ws_realtime import router as ws_realtime_router

__all__ = [
    "auth_router",
    "health_router",
    "voice_profiles_router",
    "tts_router",
    "tts_jobs_router",
    "asr_router",
    "rag_router",
    "knowledge_router",
    "chat_router",
    "metrics_router",
    "ws_realtime_router",
]
