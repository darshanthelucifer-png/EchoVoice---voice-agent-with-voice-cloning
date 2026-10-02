"""
Voice Profile Schemas (backend/app/schemas/voice_profile.py)
-----------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic models for Voice Enrollment, Audio Quality reports, and Consent Gate.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class AudioQualityMetrics(BaseModel):
    snr_db: float = Field(..., description="Signal-to-noise ratio in decibels")
    clipping_ratio: float = Field(..., description="Fraction of clipped samples")
    speech_duration_seconds: float = Field(..., description="Duration of active speech")
    speech_ratio: float = Field(..., description="Ratio of speech to total duration")
    noise_floor_db: float = Field(..., description="Estimated background noise floor")
    is_usable: bool = Field(..., description="Whether audio passes minimum enrollment thresholds")
    recommendation: str = Field(..., description="Actionable tip to improve recording")


class VoiceProfileCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=100)
    description: Optional[str] = Field(None, max_length=255)
    consent_given: bool = Field(
        ...,
        description="Explicit user confirmation that they own or have permission for this voice"
    )


class VoiceProfileRead(BaseModel):
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    reference_audio_path: str
    consent_given: bool
    consent_timestamp: Optional[datetime] = None
    quality_metrics: Optional[Dict[str, Any]] = None
    is_default: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
