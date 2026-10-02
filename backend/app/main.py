"""
Main FastAPI Application Entrypoint (backend/app/main.py)
---------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Lifespan Context Manager (`@asynccontextmanager`): Manages the application lifecycle
  (startup warm-up, DB table creation, directory verification, and shutdown cleanup)
  using modern ASGI lifespan protocols.
- Middleware Pipeline: Configures CORS headers and SlowAPI rate limiter exception handling.
- Modular Router Aggregation: Clean separation of concerns with versioned route prefixes.
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.core.config import settings
from app.core.logging import logger
from app.core.database import init_db
from app.core.rate_limit import limiter
from app.api.routes import (
    auth_router,
    health_router,
    voice_profiles_router,
    tts_router,
    tts_jobs_router,
    asr_router,
    rag_router,
    knowledge_router,
    chat_router,
    metrics_router,
    ws_realtime_router,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Asynchronous lifespan context manager for startup and shutdown events.
    
    Startup:
    1. Ensures required file directories exist (uploads, profiles, exports, checkpoints).
    2. Initializes database schema (creates tables if missing).
    3. Logs service status and loaded model configurations.
    
    Shutdown:
    1. Closes open connections and performs graceful termination.
    """
    logger.info(f"Starting {settings.APP_NAME} in [{settings.APP_ENV}] mode...")
    
    # Ensure local storage directories exist
    settings.ensure_directories()
    logger.info("Storage directories verified.")

    # Initialize DB schema
    try:
        await init_db()
        logger.info("Database connection and tables initialized.")
    except Exception as exc:
        logger.error(f"Failed to initialize database: {exc}")
        raise exc

    logger.info(f"Configured models: ASR={settings.ASR_MODEL} | LLM={settings.LLM_MODEL} | TTS={settings.TTS_ENGINE}")
    logger.info(f"{settings.APP_NAME} startup complete. Ready to receive requests.")

    yield  # Application runs here

    logger.info(f"Shutting down {settings.APP_NAME}...")


def create_application() -> FastAPI:
    """
    Application factory pattern constructing and configuring the FastAPI instance.
    """
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description=(
            "EchoVoice: Real-time conversational voice agent with RAG "
            "and studio-quality voice cloning. Powered by 100% free open-source models."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # Attach rate limiter to application state
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Root Health Check (unversioned for load balancers)
    app.include_router(health_router)

    # Versioned API Routers (/api/v1)
    app.include_router(health_router, prefix=settings.API_V1_STR)
    app.include_router(auth_router, prefix=settings.API_V1_STR)
    app.include_router(voice_profiles_router, prefix=settings.API_V1_STR)
    app.include_router(tts_router, prefix=settings.API_V1_STR)
    app.include_router(tts_jobs_router, prefix=settings.API_V1_STR)
    app.include_router(asr_router, prefix=settings.API_V1_STR)
    app.include_router(rag_router, prefix=settings.API_V1_STR)
    app.include_router(knowledge_router, prefix=settings.API_V1_STR)
    app.include_router(chat_router, prefix=settings.API_V1_STR)
    app.include_router(metrics_router, prefix=settings.API_V1_STR)
    app.include_router(ws_realtime_router, prefix=settings.API_V1_STR)

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
        origin = request.headers.get("origin")
        headers = {}
        if origin and (origin in settings.CORS_ORIGINS or "*" in settings.CORS_ORIGINS):
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
        elif settings.CORS_ORIGINS:
            headers["Access-Control-Allow-Origin"] = settings.CORS_ORIGINS[0]
            headers["Access-Control-Allow-Credentials"] = "true"

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            headers=headers,
            content={
                "success": False,
                "message": "An internal server error occurred.",
                "detail": str(exc) if settings.DEBUG else "Internal server error"
            }
        )

    return app


app = create_application()
