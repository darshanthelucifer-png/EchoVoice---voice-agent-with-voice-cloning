"""
Models Package Initialization (backend/app/models/__init__.py)
--------------------------------------------------------------
Exposes all database models so SQLAlchemy's Base metadata registers all tables.
"""

from app.models.base import BaseModelMixin
from app.models.user import User
from app.models.voice_profile import VoiceProfile
from app.models.tts_job import TTSJob, JobStatus
from app.models.conversation import Conversation, Message
from app.models.document import Document
from app.models.metric import LatencyMetric

__all__ = [
    "BaseModelMixin",
    "User",
    "VoiceProfile",
    "TTSJob",
    "JobStatus",
    "Conversation",
    "Message",
    "Document",
    "LatencyMetric",
]
