"""
Voice Profile Model (backend/app/models/voice_profile.py)
---------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Foreign Keys & Integrity: Enforces relational integrity referencing `users.id`.
- Consent Auditing: Mandatory storage of ethical consent flag and consent timestamp.
- JSON Type Handling: Stores flexible audio quality analysis (SNR, clipping, duration)
  directly in database columns.
"""

from sqlalchemy import Column, String, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.base import BaseModelMixin


class VoiceProfile(Base, BaseModelMixin):
    """
    Stores an enrolled voice profile consisting of clean audio references,
    speaker embeddings, quality metrics, and ethical consent records.
    """
    __tablename__ = "voice_profiles"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    description = Column(String(255), nullable=True)

    # File paths to processed audio and extracted embeddings
    reference_audio_path = Column(String(500), nullable=False)
    raw_audio_path = Column(String(500), nullable=True)
    speaker_embedding_path = Column(String(500), nullable=True)

    # Consent gate (Mandatory for ethical voice cloning)
    consent_given = Column(Boolean, default=False, nullable=False)
    consent_timestamp = Column(DateTime(timezone=True), nullable=True)

    # Audio quality metrics (SNR, clipping, speech ratio, duration)
    quality_metrics = Column(JSON, nullable=True)
    is_default = Column(Boolean, default=False, nullable=False)

    # Relationships
    user = relationship("User", back_populates="voice_profiles")
    tts_jobs = relationship("TTSJob", back_populates="voice_profile")

    def __repr__(self) -> str:
        return f"<VoiceProfile id={self.id} name={self.name} user_id={self.user_id}>"
