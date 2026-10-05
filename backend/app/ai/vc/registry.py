"""
Voice Conversion Engine Registry (backend/app/ai/vc/registry.py)
----------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Registry Pattern: Central repository mapping voice conversion algorithm identifiers
  ("seed_vc", "tier2_seed_vc", "rvc_v2") to engine implementations.
- Factory Pattern: Decouples high-level service requests from concrete converter
  instantiation, allowing models to be swapped via configuration.
- Single Responsibility Principle (SRP): Registry solely manages engine lifecycle
  and lookup.
"""

from typing import Dict, List, Optional
from app.core.config import settings
from app.core.logging import logger
from app.ai.vc.base import VoiceConverterEngine
from app.ai.vc.seed_vc_engine import seed_vc_engine, SeedVCEngine
from app.ai.vc.rvc_engine import rvc_engine, RVCInference


class VCRegistry:
    """
    Central registry for Voice Conversion (VC) engines.
    """

    def __init__(self):
        self._engines: Dict[str, VoiceConverterEngine] = {}
        self._register_default_engines()

    def _register_default_engines(self) -> None:
        """Registers the built-in voice conversion engines."""
        self.register("seed_vc", seed_vc_engine)
        self.register("tier2_seed_vc", seed_vc_engine)
        self.register("plachta_seed_vc", seed_vc_engine)
        self.register("rvc_v2", rvc_engine)
        self.register("tier3_rvc", rvc_engine)
        self.register("tier3_rvc_v2", rvc_engine)

    def register(self, name: str, engine: VoiceConverterEngine) -> None:
        """Registers a new voice converter engine instance."""
        self._engines[name.lower()] = engine
        logger.debug(f"Registered VC engine: '{name}' -> {engine.__class__.__name__}")

    def get(self, name: Optional[str] = None) -> VoiceConverterEngine:
        """
        Retrieves a registered voice converter engine by name.
        Falls back to 'seed_vc' if name is not found or not specified.
        """
        target = (name or "seed_vc").lower()
        if target in self._engines:
            return self._engines[target]

        logger.warning(f"VC engine '{target}' not registered. Falling back to 'seed_vc'.")
        return self._engines.get("seed_vc", seed_vc_engine)

    def available_engines(self) -> List[str]:
        """Returns the list of currently registered engine identifiers."""
        return list(self._engines.keys())


# Global singleton instance
vc_registry = VCRegistry()


def get_vc_engine(name: Optional[str] = None) -> VoiceConverterEngine:
    """Helper function to retrieve a voice converter from the global registry."""
    return vc_registry.get(name)
