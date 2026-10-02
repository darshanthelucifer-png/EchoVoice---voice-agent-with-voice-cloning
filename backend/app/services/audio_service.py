"""
Audio Service (backend/app/services/audio_service.py)
-----------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Thread Pool Offloading (`asyncio.to_thread`): Offloads heavy CPU-bound signal processing
  (FFT filters, spectral gating, VAD passes) off FastAPI's async event loop into
  worker threads, ensuring the web server remains responsive to concurrent requests.
- Asynchronous File Operations: Non-blocking disk persistence with pathlib and soundfile.
- Service Layer Pattern: Encapsulates all audio business logic away from API endpoints.
"""

import asyncio
from pathlib import Path
from typing import Union, Tuple, Optional
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.ai.audio.cleanup import (
    AudioCleanupPipeline,
    CleanupConfig,
    CleanupResult,
    cleanup_pipeline,
)
from app.ai.audio.quality import AudioQualityReport, quality_analyzer


class AudioService:
    """
    Orchestrates audio analysis, voice profile cleaning, and disk caching
    for the Voice Studio module.
    """

    async def analyze_audio(
        self,
        audio_source: Union[str, Path, bytes, np.ndarray],
        sr: Optional[int] = None
    ) -> AudioQualityReport:
        """
        Non-blocking audio quality assessment offloaded to a thread worker.
        """
        def _sync_analyze():
            audio, sample_rate = AudioCleanupPipeline.load_audio(audio_source, sr)
            return quality_analyzer.analyze(audio, sample_rate)

        return await asyncio.to_thread(_sync_analyze)

    async def clean_voice_sample(
        self,
        input_source: Union[str, Path, bytes, np.ndarray],
        output_filename: Optional[str] = None,
        config: Optional[CleanupConfig] = None
    ) -> Tuple[Path, CleanupResult]:
        """
        Executes full cleanup pipeline in a worker thread and persists the cleaned
        reference audio file to the configured voice profiles directory.
        """
        cfg = config or CleanupConfig()

        def _sync_clean():
            audio, sr = AudioCleanupPipeline.load_audio(input_source)
            result = cleanup_pipeline.process(audio, sr, cfg)

            # Determine output path
            fname = output_filename or "reference.wav"
            out_path = settings.VOICE_PROFILES_DIR / fname
            AudioCleanupPipeline.save_audio(result.audio, result.sample_rate, out_path)
            result.output_path = out_path
            return out_path, result

        return await asyncio.to_thread(_sync_clean)


audio_service = AudioService()
