"""
ASR Engine Registry & Factory (backend/app/ai/asr/registry.py)
--------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Registry / Factory Pattern: Decouples high-level services and WebSocket handlers
  from specific neural inference backend implementations.
- Fallback Resolution: Gracefully defaults to fallback engine if heavy model fails to initialize.
"""

from typing import Dict, Optional, Type
from app.core.config import settings
from app.core.logging import logger
from app.ai.asr.base import ASREngine
from app.ai.asr.faster_whisper_engine import FasterWhisperEngine
from app.ai.asr.fallback_engine import LocalFallbackASREngine


class ASRRegistry:
    """Registry maintaining active and available Speech-to-Text engines."""

    def __init__(self):
        self._classes: Dict[str, Type[ASREngine]] = {}
        self._instances: Dict[str, ASREngine] = {}

        # Register default engines
        self.register_class("faster_whisper", FasterWhisperEngine)
        self.register_class("whisper", FasterWhisperEngine)
        self.register_class("fallback", LocalFallbackASREngine)

    def register_class(self, name: str, cls: Type[ASREngine]) -> None:
        self._classes[name.lower()] = cls

    def register_instance(self, name: str, instance: ASREngine) -> None:
        self._instances[name.lower()] = instance

    def get_engine(self, engine_name: Optional[str] = None) -> ASREngine:
        """Resolves requested engine or falls back to system configured engine."""
        name = (engine_name or "faster_whisper").lower().replace("-", "_")

        if name in self._instances:
            return self._instances[name]

        if name in self._classes:
            try:
                engine = self._classes[name]()
                self._instances[name] = engine
                return engine
            except Exception as exc:
                logger.warning(
                    f"Failed to initialize ASR engine '{name}': {exc}. Falling back to 'fallback'."
                )

        # Fallback to local deterministic engine
        if "fallback" not in self._instances:
            self._instances["fallback"] = LocalFallbackASREngine()
        return self._instances["fallback"]

    def available_engines(self) -> Dict[str, str]:
        """Lists all registered ASR engine identifiers."""
        return {k: v.__name__ for k, v in self._classes.items()}


asr_registry = ASRRegistry()


def get_asr_engine(engine_name: Optional[str] = None) -> ASREngine:
    """Helper shortcut returning configured ASR engine instance."""
    return asr_registry.get_engine(engine_name)
