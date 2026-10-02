"""
TTS Job Schemas (backend/app/schemas/tts_job.py)
------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic models with validation for long-form speech generation requests,
  checkpoint progress updates, and audio reports.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from app.models.tts_job import JobStatus


class TTSJobCreate(BaseModel):
    voice_profile_id: Optional[str] = None
    script_text: str = Field(..., min_length=1, description="Script text to generate speech for")
    target_language: str = Field(default="en", max_length=10)
    engine: str = Field(default="xtts_v2")
    mastering_preset: str = Field(default="youtube_voiceover")
    auto_translate: bool = False


class TTSJobRead(BaseModel):
    id: str
    user_id: str
    voice_profile_id: Optional[str] = None
    status: JobStatus
    target_language: str
    engine: str
    mastering_preset: str
    progress: float
    total_chunks: int
    completed_chunks: int
    output_wav_path: Optional[str] = None
    output_mp3_path: Optional[str] = None
    output_m4a_path: Optional[str] = None
    wer_score: Optional[float] = None
    similarity_score: Optional[float] = None
    audio_report: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
