"""
Voice Conversion Subpackage (backend/app/ai/vc/__init__.py)
"""

from app.ai.vc.base import VoiceConverterEngine, VCOutput
from app.ai.vc.seed_vc_engine import SeedVCEngine, seed_vc_engine
from app.ai.vc.rvc_engine import RVCInference, rvc_engine
from app.ai.vc.registry import VCRegistry, vc_registry, get_vc_engine

__all__ = [
    "VoiceConverterEngine",
    "VCOutput",
    "SeedVCEngine",
    "seed_vc_engine",
    "RVCInference",
    "rvc_engine",
    "VCRegistry",
    "vc_registry",
    "get_vc_engine",
]
