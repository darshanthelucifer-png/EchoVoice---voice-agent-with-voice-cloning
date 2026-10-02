"""
TTS Job Model (backend/app/models/tts_job.py)
---------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- State Machine Pattern: Tracks life-cycle status (PENDING -> PROCESSING -> COMPLETED/FAILED/CANCELLED).
- Checkpointing tracking: Records completed chunk count, total chunks, and checkpoint
  directory for resilient long-form (up to 1 hour) speech generation.
"""

from enum import Enum
from sqlalchemy import Column, String, Float, Integer, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.base import BaseModelMixin


class JobStatus(str, Enum):
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TTSJob(Base, BaseModelMixin):
    """
    Represents an asynchronous speech synthesis job (especially for long-form generation).
    Supports checkpointing, progress tracking, and quality metrics.
    """
    __tablename__ = "tts_jobs"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    voice_profile_id = Column(String(36), ForeignKey("voice_profiles.id", ondelete="SET NULL"), nullable=True)

    # Job Configuration
    status = Column(String(20), default=JobStatus.PENDING.value, nullable=False, index=True)
    script_text = Column(Text, nullable=False)
    target_language = Column(String(10), default="en", nullable=False)
    engine = Column(String(50), default="xtts_v2", nullable=False)
    mastering_preset = Column(String(50), default="youtube_voiceover", nullable=False)

    # Checkpoint and Progress Monitoring
    progress = Column(Float, default=0.0, nullable=False)  # 0.0 to 100.0%
    total_chunks = Column(Integer, default=0, nullable=False)
    completed_chunks = Column(Integer, default=0, nullable=False)
    checkpoint_dir = Column(String(500), nullable=True)

    # Exported Files (YouTube standard)
    output_wav_path = Column(String(500), nullable=True)
    output_mp3_path = Column(String(500), nullable=True)
    output_m4a_path = Column(String(500), nullable=True)

    # Quality Assurance Metrics
    wer_score = Column(Float, nullable=True)
    similarity_score = Column(Float, nullable=True)
    audio_report = Column(JSON, nullable=True)  # LUFS, true peak, LRA, noise floor
    error_message = Column(Text, nullable=True)

    # Relationships
    user = relationship("User", back_populates="tts_jobs")
    voice_profile = relationship("VoiceProfile", back_populates="tts_jobs")

    def __repr__(self) -> str:
        return f"<TTSJob id={self.id} status={self.status} progress={self.progress}%>"
