"""
TTS Package Initialization (backend/app/ai/tts/__init__.py)
-----------------------------------------------------------
Exports the unified TTSEngine interface, XTTS-v2 engine, fallbacks, and registry.
"""

from app.ai.tts.base import TTSEngine, TTSOutput, TTSConfig
from app.ai.tts.fallback_engine import LocalFallbackEngine
from app.ai.tts.xtts_engine import XTTSEngine
from app.ai.tts.registry import TTSRegistry, tts_registry, get_tts_engine

__all__ = [
    "TTSEngine",
    "TTSOutput",
    "TTSConfig",
    "LocalFallbackEngine",
    "XTTSEngine",
    "TTSRegistry",
    "tts_registry",
    "get_tts_engine",
]
