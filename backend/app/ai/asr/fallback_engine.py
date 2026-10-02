"""
Local Fallback ASR Engine (backend/app/ai/asr/fallback_engine.py)
-----------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Graceful Degradation / Fallback Strategy: Ensures the application remains fully functional
  and testable even in lightweight development environments without neural model weights.
"""

from pathlib import Path
from typing import List, Optional, Union
import numpy as np

from app.core.logging import logger, timed_step
from app.ai.asr.base import ASREngine, ASRResult, ASRToken


class LocalFallbackASREngine(ASREngine):
    """
    Lightweight, deterministic ASR engine for offline testing and continuous integration.
    """

    engine_name: str = "fallback"

    def __init__(self, default_text: Optional[str] = None):
        self.default_text = default_text or "Hello, I am testing the EchoVoice conversational speech recognition system."

    @timed_step("Local Fallback ASR Decoding")
    async def transcribe(
        self,
        audio: np.ndarray,
        sr: int = 16_000,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """Generates deterministic ASR telemetry and mock tokens aligned with audio duration."""
        duration_sec = len(audio) / float(sr)
        text = self.default_text if duration_sec >= 0.5 else ""

        words = text.split()
        tokens: List[ASRToken] = []

        if words and duration_sec > 0:
            step = duration_sec / len(words)
            for i, w in enumerate(words):
                tokens.append(
                    ASRToken(
                        word=w,
                        start_sec=round(i * step, 3),
                        end_sec=round((i + 1) * step, 3),
                        probability=0.98
                    )
                )

        return ASRResult(
            text=text,
            language=language or "en",
            duration_seconds=round(duration_sec, 3),
            latency_ms=12.5,
            rtf=0.015,
            words=tokens,
            confidence=0.98,
            engine_name="fallback-local"
        )

    async def transcribe_file(
        self,
        file_path: Union[str, Path],
        language: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """Transcribes audio file with fallback engine."""
        import soundfile as sf
        data, sr = sf.read(str(file_path), dtype="float32")
        return await self.transcribe(data, sr=sr, language=language, word_timestamps=word_timestamps)
