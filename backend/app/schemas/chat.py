"""
Conversational Chat Schemas (backend/app/schemas/chat.py)
---------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic v2 Models: Request validation and serialization for multi-turn conversations,
  chat turns, and SSE streaming telemetry.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MessageSchema(BaseModel):
    """Schema for an individual conversational message turn."""
    id: str
    role: str
    content: str
    emotion_tag: Optional[str] = None
    audio_path: Optional[str] = None
    latency_ms: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class ConversationCreate(BaseModel):
    """Request payload to initiate a new multi-turn conversation."""
    title: Optional[str] = Field(default="New Voice Conversation", max_length=200)
    persona: str = Field(default="friendly_assistant", description="Conversational voice persona")


class ConversationRead(BaseModel):
    """Full conversation history representation."""
    id: str
    user_id: str
    title: str
    persona: str
    created_at: datetime
    messages: List[MessageSchema] = Field(default_factory=list)

    model_config = {"from_attributes": True}


class ChatCompletionRequest(BaseModel):
    """Request payload for interacting with the conversational AI agent."""
    message: str = Field(..., min_length=1, max_length=2000, description="User speech transcript or text input")
    conversation_id: Optional[str] = Field(None, description="Optional conversation session ID")
    persona: str = Field(default="friendly_assistant", description="Conversational persona")
    engine: Optional[str] = Field(None, description="LLM backend override ('fallback', 'ollama', 'openai_compatible')")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=256, ge=16, le=1024)
    stream: bool = Field(default=False, description="Enable Server-Sent Events (SSE) streaming")
    use_rag: bool = Field(default=True, description="Query knowledge base for factual grounding")
    collection_name: str = Field(default="default", description="RAG collection partition to query")


class ChatCompletionResponse(BaseModel):
    """Synchronous chat turn response with latency and RAG provenance."""
    message: str
    conversation_id: str
    model_name: str
    latency_ms: float
    rag_sources: List[str] = Field(default_factory=list)
