"""
Base Voice Conversion Engine Interface (backend/app/ai/vc/base.py)
------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Abstract Base Classes (`abc.ABC`, `@abstractmethod`): Enforces a strict, unified
  contract across all voice conversion engines (Seed-VC, RVC v2, CosyVoice-VC).
- Dataclasses (`@dataclass`): Standardized, immutable data carriers containing
  converted waveforms, biometric verification scores, latency measurements,
  and gate decision telemetry.
- Strict Type Hinting: Full static typing with Union, Optional, Tuple, and Dict
  for robust ML pipeline orchestration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import numpy as np


@dataclass
class VCOutput:
    """
    Standardized voice conversion payload returned by any VoiceConverterEngine.
    """
    audio: np.ndarray                            # Converted 1D float32 waveform
    sample_rate: int                             # e.g., 24000 Hz or 48000 Hz
    duration_seconds: float                      # Duration in seconds
    latency_ms: float                            # Conversion time in milliseconds
    engine_name: str                             # "seed_vc", "rvc_v2", etc.
    source_reference: Optional[str] = None       # Identifier of source audio
    target_reference: Optional[str] = None       # Identifier of target speaker voice
    likeness_score: Optional[float] = None       # Biometric similarity score (0.0 to 1.0)
    passed_gate: bool = True                     # True if likeness >= target gate
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-Time Factor: latency_seconds / audio_duration_seconds (< 1.0 is faster than real time)."""
        if self.duration_seconds <= 0:
            return 0.0
        return (self.latency_ms / 1000.0) / self.duration_seconds


class VoiceConverterEngine(ABC):
    """
    Abstract Base Class defining the contract for Voice Conversion (VC) engines.
    Adheres to the Strategy Pattern to allow seamless swapping between Seed-VC,
    RVC v2, and remote zero-GPU Hugging Face Spaces.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Unique identifier for this voice conversion engine."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the engine's model weights or remote endpoint are ready."""
        pass

    @abstractmethod
    async def convert_voice(
        self,
        source_audio: Union[np.ndarray, str, Path],
        target_reference: Union[np.ndarray, str, Path],
        source_sr: int = 24000,
        target_sr: int = 24000,
        diffusion_steps: int = 10,
        f0_condition: bool = True,
        **kwargs: Any
    ) -> VCOutput:
        """
        Converts the timbre of `source_audio` into the vocal identity of `target_reference`
        while preserving pitch, rhythm, and linguistic content.
        """
        pass
