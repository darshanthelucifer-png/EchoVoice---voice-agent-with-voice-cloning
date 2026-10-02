"""
ASR Service (backend/app/services/asr_service.py)
-------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Service Facade Pattern: Coordinates multi-format audio decoding (WAV, MP3, FLAC, WebM),
  sample rate conversion, engine resolution, and streaming session lifecycle.
"""

from pathlib import Path
from typing import Optional, Union, Tuple
import io
import numpy as np
import soundfile as sf

from app.core.logging import logger, timed_step
from app.ai.audio.cleanup import AudioCleanupPipeline
from app.ai.asr.base import ASRResult
from app.ai.asr.registry import get_asr_engine
from app.ai.asr.stream import VADAudioStreamProcessor, StreamConfig


class ASRService:
    """High-level service coordinating speech-to-text inference and real-time streaming."""

    @timed_step("ASR Audio Bytes Transcription")
    async def transcribe_audio_bytes(
        self,
        audio_bytes: bytes,
        language: Optional[str] = None,
        engine_name: Optional[str] = None,
        word_timestamps: bool = True
    ) -> ASRResult:
        """
        Decodes incoming audio bytes (WAV, MP3, WebM, OGG), resamples to 16 kHz,
        and executes speech recognition.
        """
        # Load and convert to mono float32
        audio, sr = AudioCleanupPipeline.load_audio(audio_bytes)

        engine = get_asr_engine(engine_name)
        result = await engine.transcribe(
            audio=audio,
            sr=sr,
            language=language,
            word_timestamps=word_timestamps
        )
        return result

    def create_stream_processor(
        self,
        engine_name: Optional[str] = None,
        config: Optional[StreamConfig] = None
    ) -> VADAudioStreamProcessor:
        """Creates a stateful VAD-gated audio streaming processor for WebSocket sessions."""
        engine = get_asr_engine(engine_name)
        return VADAudioStreamProcessor(engine=engine, config=config)


asr_service = ASRService()
