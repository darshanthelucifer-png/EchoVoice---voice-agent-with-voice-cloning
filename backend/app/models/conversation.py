"""
Conversation Models (backend/app/models/conversation.py)
---------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Relational mapping between Conversations and Messages.
- Structured storage for conversational metadata: role, emotion tag, audio path,
  and end-to-end turn latency.
"""

from sqlalchemy import Column, String, Float, ForeignKey, Text
from sqlalchemy.orm import relationship
from app.core.database import Base
from app.models.base import BaseModelMixin


class Conversation(Base, BaseModelMixin):
    """
    Groups real-time voice and text chat turns.
    """
    __tablename__ = "conversations"

    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String(200), default="New Voice Conversation", nullable=False)
    persona = Column(String(50), default="friendly_assistant", nullable=False)

    # Relationships
    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
        lazy="selectin"
    )

    def __repr__(self) -> str:
        return f"<Conversation id={self.id} title={self.title}>"


class Message(Base, BaseModelMixin):
    """
    Represents an individual conversational turn (user speech or assistant response).
    """
    __tablename__ = "messages"

    conversation_id = Column(String(36), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False, index=True)
    role = Column(String(20), nullable=False)  # "user", "assistant", "system"
    content = Column(Text, nullable=False)
    audio_path = Column(String(500), nullable=True)
    emotion_tag = Column(String(50), nullable=True)  # "calm", "warm", "excited", "serious"
    latency_ms = Column(Float, nullable=True)  # End-to-end turn latency

    # Relationship
    conversation = relationship("Conversation", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role}>"
