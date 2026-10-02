"""
ASR Module exports (backend/app/ai/asr/__init__.py)
"""

from app.ai.asr.base import ASREngine, ASRResult, ASRToken
from app.ai.asr.faster_whisper_engine import FasterWhisperEngine
from app.ai.asr.fallback_engine import LocalFallbackASREngine
from app.ai.asr.registry import asr_registry, get_asr_engine
from app.ai.asr.stream import VADAudioStreamProcessor, StreamConfig, StreamEventType

__all__ = [
    "ASREngine",
    "ASRResult",
    "ASRToken",
    "FasterWhisperEngine",
    "LocalFallbackASREngine",
    "asr_registry",
    "get_asr_engine",
    "VADAudioStreamProcessor",
    "StreamConfig",
    "StreamEventType",
]
