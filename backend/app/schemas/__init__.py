"""
Schemas Package Initialization (backend/app/schemas/__init__.py)
----------------------------------------------------------------
Exports all request/response validation schemas.
"""

from app.schemas.common import APIResponse, HealthResponse
from app.schemas.user import UserBase, UserCreate, UserUpdate, UserRead
from app.schemas.auth import Token, TokenPayload, LoginRequest
from app.schemas.voice_profile import (
    AudioQualityMetrics,
    VoiceProfileCreate,
    VoiceProfileRead,
)
from app.schemas.tts_job import TTSJobCreate, TTSJobRead

__all__ = [
    "APIResponse",
    "HealthResponse",
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserRead",
    "Token",
    "TokenPayload",
    "LoginRequest",
    "AudioQualityMetrics",
    "VoiceProfileCreate",
    "VoiceProfileRead",
    "TTSJobCreate",
    "TTSJobRead",
]
