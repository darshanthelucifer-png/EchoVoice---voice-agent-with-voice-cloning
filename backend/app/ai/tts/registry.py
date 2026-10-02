"""
TTS Engine Registry (backend/app/ai/tts/registry.py)
----------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Strategy Pattern & Registry / Factory Pattern: Central lookup registry where
  TTS engines are registered and resolved by string key.
- Config-driven Swapping: Resolves engine specified in `settings.TTS_ENGINE`
  without requiring code modifications.
- Singleton Cache: Avoids re-instantiating heavy neural network or API client wrappers.
"""

from typing import Dict, Type, Optional, List
from app.core.config import settings
from app.core.logging import logger
from app.ai.tts.base import TTSEngine
from app.ai.tts.xtts_engine import XTTSEngine
from app.ai.tts.fallback_engine import LocalFallbackEngine


class TTSRegistry:
    """
    Registry factory mapping engine names to instantiated TTSEngine strategies.
    """
    def __init__(self):
        self._engines: Dict[str, TTSEngine] = {}
        self._classes: Dict[str, Type[TTSEngine]] = {}

        # Register default engines
        self.register_class("xtts_v2", XTTSEngine)
        self.register_class("fallback", LocalFallbackEngine)
        self.register_class("local_fallback", LocalFallbackEngine)

    def register_class(self, name: str, cls: Type[TTSEngine]) -> None:
        """Registers an engine class under a unique identifier."""
        self._classes[name.lower()] = cls

    def register_instance(self, name: str, instance: TTSEngine) -> None:
        """Registers a pre-instantiated engine."""
        self._engines[name.lower()] = instance

    def get_engine(self, engine_name: Optional[str] = None) -> TTSEngine:
        """
        Resolves the requested engine or falls back to the configured default in settings.
        """
        name = (engine_name or settings.TTS_ENGINE).lower()

        if name in self._engines:
            return self._engines[name]

        if name in self._classes:
            engine_instance = self._classes[name]()
            self._engines[name] = engine_instance
            return engine_instance

        logger.warning(f"Engine '{name}' not found in registry. Falling back to 'fallback' engine.")
        return self.get_engine("fallback")

    def available_engines(self) -> List[str]:
        """Lists all registered engine identifiers."""
        return list(set(list(self._classes.keys()) + list(self._engines.keys())))


# Global registry singleton
tts_registry = TTSRegistry()


def get_tts_engine(name: Optional[str] = None) -> TTSEngine:
    """Convenience helper to retrieve the active TTSEngine."""
    return tts_registry.get_engine(name)
