"""
ASR Abstract Base Class & Data Models (backend/app/ai/asr/base.py)
------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Abstract Base Class (`abc.ABC`): Formalizes the contract for speech recognition engines.
- Dataclasses: Strictly typed token and transcription results with timing, confidence,
  and Real-Time Factor (RTF) telemetry.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Union
import numpy as np


@dataclass(frozen=True)
class ASRToken:
    """Individual word or token with start/end boundary timestamps and confidence."""
    word: str
    start_sec: float
    end_sec: float
    probability: float


@dataclass
class ASRResult:
    """Standardized result returned by all speech-to-text engines."""
    text: str
    language: str
    duration_seconds: float
    latency_ms: float
    rtf: float                              # Real-Time Factor: latency_seconds / audio_duration
    words: List[ASRToken] = field(default_factory=list)
    confidence: float = 1.0
    engine_name: str = "unknown"

    @property
    def is_empty(self) -> bool:
        return not self.text.strip()


class ASREngine(ABC):
    """Abstract Base Class for pluggable Speech-to-Text inference backends."""

    engine_name: str = "base"

    @abstractmethod
    async def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """
        Transcribes a 1D float32 audio array into text with word timings.

        Args:
            audio: 1D float32 numpy array, normalized to [-1.0, 1.0].
            sr: Sample rate in Hz (typically 16,000 for Whisper).
            language: Optional ISO 639-1 code (e.g. 'en', 'es', 'hi'). None = auto-detect.
            prompt: Optional preceding text prompt to condition vocabulary or technical terms.
            word_timestamps: Whether to extract token-level start and end times.
        """
        pass

    @abstractmethod
    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """Transcribes an audio file on disk."""
        pass
