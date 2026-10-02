"""
User Model (backend/app/models/user.py)
---------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- SQLAlchemy ORM Relationships: Demonstrates 1-to-many relationship mapping
  (`relationship`) between Users, VoiceProfiles, and TTSJobs with cascade deletes.
- Column constraints: Unique indexes, nullable constraints, and boolean flags.
"""

from sqlalchemy import Column, String, Boolean
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.base import BaseModelMixin


class User(Base, BaseModelMixin):
    """
    Represents an authenticated system user.
    """
    __tablename__ = "users"

    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    is_superuser = Column(Boolean, default=False, nullable=False)

    # Relationships
    voice_profiles = relationship(
        "VoiceProfile",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    tts_jobs = relationship(
        "TTSJob",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    conversations = relationship(
        "Conversation",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )
    documents = relationship(
        "Document",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email}>"
