"""
Coqui XTTS-v2 Engine (backend/app/ai/tts/xtts_engine.py)
-------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Strategy & Multi-Backend Dispatch: Tries local neural model first, falls back to
  free Hugging Face Space via `gradio_client`, and finally to `LocalFallbackEngine`
  if offline or unconfigured.
- Thread Pool Execution (`asyncio.to_thread`): Prevents deep learning inference
  from blocking the FastAPI async event loop.
- Sentence-level Streaming Generator: Yields low-latency audio segments as each
  sentence completes synthesis.
"""

import time
import asyncio
from pathlib import Path
from typing import Optional, List, Union, Any, AsyncGenerator
import numpy as np

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.audio.cleanup import AudioCleanupPipeline
from app.ai.tts.base import TTSEngine, TTSOutput
from app.ai.tts.fallback_engine import LocalFallbackEngine


class XTTSEngine(TTSEngine):
    """
    Coqui XTTS-v2 Zero-Shot Voice Cloning Engine.
    Clones any speaker timbre from a 6-20 second clean reference clip across 17 languages.
    """

    XTTS_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr",
        "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
    ]

    def __init__(self, hf_space: Optional[str] = None):
        self._local_model = None
        self._hf_space = hf_space or "coqui/xtts"
        self._fallback = LocalFallbackEngine()

    @property
    def engine_name(self) -> str:
        return "xtts_v2"

    def is_available(self) -> bool:
        return True

    def is_voice_cloning_supported(self) -> bool:
        return True

    def supported_languages(self) -> List[str]:
        return self.XTTS_LANGUAGES

    def _synthesize_via_local_coqui(
        self,
        text: str,
        speaker_wav: Union[str, Path],
        language: str,
        speed: float = 1.0
    ) -> Optional[np.ndarray]:
        """Synthesize using local Coqui TTS library if installed."""
        try:
            from TTS.api import TTS
            if self._local_model is None:
                device = "cuda" if getattr(settings, "TTS_DEVICE", "cpu") == "cuda" else "cpu"
                self._local_model = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

            wav = self._local_model.tts(
                text=text,
                speaker_wav=str(speaker_wav),
                language=language,
                speed=speed
            )
            return np.array(wav, dtype=np.float32)
        except Exception as e:
            logger.debug(f"Local Coqui XTTS not active: {e}")
            return None

    def _synthesize_via_hf_space(
        self,
        text: str,
        speaker_wav: Union[str, Path],
        language: str,
        speed: float = 1.0
    ) -> Optional[np.ndarray]:
        """Synthesize using a free Hugging Face Space via gradio_client."""
        try:
            from gradio_client import Client, handle_file
            token = settings.HF_TOKEN if settings.HF_TOKEN else None
            client = Client(self._hf_space, token=token)

            # XTTS-v2 Gradio standard parameters: (text, language, audio_file, ...)
            result = client.predict(
                prompt=text,
                language=language,
                audio_file_pth=handle_file(str(speaker_wav)),
                api_name="/predict"
            )
            if result and Path(result).exists():
                audio, _ = AudioCleanupPipeline.load_audio(result, target_sr=24000)
                return audio
        except Exception as e:
            logger.debug(f"HF Space XTTS call note: {e}")
            return None

    @timed_step("XTTS-v2 Voice Synthesis")
    def _synthesize_sync(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None
    ) -> TTSOutput:
        start_time = time.perf_counter()
        sr = 24000
        audio_arr: Optional[np.ndarray] = None
        method_used = "fallback"

        # Normalize language tag (e.g. "zh" -> "zh-cn")
        lang = "zh-cn" if language.lower() in ["zh", "chinese", "zh-cn"] else language.lower()
        if lang not in self.XTTS_LANGUAGES:
            logger.info(f"Language '{language}' not directly in XTTS-v2 17 native set; using 'en' voice base.")
            lang = "en"

        # 1. Attempt Local Coqui XTTS if speaker wav is provided
        if speaker_wav and Path(speaker_wav).exists():
            audio_arr = self._synthesize_via_local_coqui(text, speaker_wav, lang, speed)
            if audio_arr is not None:
                method_used = "local_coqui"

        # 2. Attempt Remote Hugging Face Space
        if audio_arr is None and speaker_wav and Path(speaker_wav).exists():
            audio_arr = self._synthesize_via_hf_space(text, speaker_wav, lang, speed)
            if audio_arr is not None:
                method_used = "hf_space_gradio"

        # 3. Graceful Fallback
        if audio_arr is None:
            fallback_out = self._fallback._synthesize_sync(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                speed=speed,
                emotion=emotion
            )
            audio_arr = fallback_out.audio
            method_used = "local_fallback"

        duration = len(audio_arr) / sr
        latency = (time.perf_counter() - start_time) * 1000

        return TTSOutput(
            audio=audio_arr,
            sample_rate=sr,
            duration_seconds=round(duration, 2),
            latency_ms=round(latency, 2),
            engine_name="xtts_v2",
            language=lang,
            speaker_reference=str(speaker_wav) if speaker_wav else None,
            metadata={"backend": method_used, "emotion": emotion, "speed": speed}
        )

    async def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        **kwargs: Any
    ) -> TTSOutput:
        return await asyncio.to_thread(
            self._synthesize_sync,
            text,
            speaker_wav,
            language,
            speed,
            emotion
        )

    async def synthesize_stream(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        **kwargs: Any
    ) -> AsyncGenerator[bytes, None]:
        """Sentence-level streaming generator for low Time-to-First-Audio (TTFA)."""
        import re
        sentences = [s.strip() for s in re.split(r"[.!?।\n]+", text) if s.strip()]
        if not sentences:
            sentences = [text]

        for sentence in sentences:
            output = await self.synthesize(
                sentence,
                speaker_wav=speaker_wav,
                language=language,
                speed=speed,
                emotion=emotion
            )
            pcm_data = (output.audio * 32767.0).astype(np.int16).tobytes()
            yield pcm_data
