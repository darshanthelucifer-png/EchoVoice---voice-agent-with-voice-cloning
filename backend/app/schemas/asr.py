"""
ASR Schemas (backend/app/schemas/asr.py)
----------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic v2 Models: Validates speech recognition outputs and WebSocket telemetry payloads.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class ASRTokenSchema(BaseModel):
    word: str
    start_sec: float
    end_sec: float
    probability: float


class ASRResponse(BaseModel):
    text: str
    language: str
    duration_seconds: float
    latency_ms: float
    rtf: float
    confidence: float
    engine_name: str
    words: List[ASRTokenSchema] = Field(default_factory=list)
