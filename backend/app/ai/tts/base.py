"""
Base TTS Engine Interface (backend/app/ai/tts/base.py)
------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Abstract Base Classes (`abc.ABC`, `@abstractmethod`): Enforces a unified,
  strict contract across all pluggable text-to-speech and voice cloning engines
  (XTTS-v2, OpenVoiceV2, F5-TTS, Parler, and local fallbacks).
- Dataclasses (`@dataclass`): Clean, immutable data transfer objects carrying
  synthesized float32 waveforms, sample rates, latency metrics, and engine metadata.
- Asynchronous Generators (`AsyncGenerator`): Streaming audio chunks in real-time
  to achieve low Time-To-First-Audio (TTFA) for voice assistants.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
import numpy as np


@dataclass
class TTSOutput:
    """
    Standardized synthesis payload returned by any TTSEngine.
    """
    audio: np.ndarray                       # Float32 1D numpy array
    sample_rate: int                        # e.g., 24000 Hz or 22050 Hz
    duration_seconds: float                 # Audio duration
    latency_ms: float                       # Total inference time in milliseconds
    engine_name: str                        # "xtts_v2", "openvoice", etc.
    language: str                           # "en", "es", "hi", etc.
    speaker_reference: Optional[str] = None # Path/ID of cloned voice reference
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-Time Factor: latency_seconds / audio_duration_seconds (< 1.0 is faster than real time)."""
        if self.duration_seconds <= 0:
            return 0.0
        return (self.latency_ms / 1000.0) / self.duration_seconds


@dataclass
class TTSConfig:
    """
    Inference parameters for speech generation.
    """
    language: str = "en"
    speed: float = 1.0
    temperature: float = 0.75
    repetition_penalty: float = 2.0
    emotion: Optional[str] = None           # "calm", "warm", "excited", "serious"
    stream_chunk_size: int = 20             # Token / word chunk size for streaming


class TTSEngine(ABC):
    """
    Abstract Base Class defining the unified interface for all EchoVoice TTS engines.
    Follows the Strategy Pattern so engines can be swapped purely via configuration.
    """

    @property
    @abstractmethod
    def engine_name(self) -> str:
        """Unique identifier for this engine (e.g. 'xtts_v2')."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Returns True if the engine's model or remote endpoint is ready for inference."""
        pass

    @abstractmethod
    def is_voice_cloning_supported(self) -> bool:
        """Returns True if this engine supports zero-shot speaker conditioning."""
        pass

    @abstractmethod
    def supported_languages(self) -> List[str]:
        """Returns list of ISO language codes supported by this engine."""
        pass

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        **kwargs: Any
    ) -> TTSOutput:
        """
        Synthesizes speech from text, optionally cloning the timbre in `speaker_wav`.
        """
        pass

    @abstractmethod
    async def synthesize_stream(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[bytes, None]:
        """
        Yields raw PCM audio chunk bytes as they are generated to enable low-latency playback.
        """
        pass
