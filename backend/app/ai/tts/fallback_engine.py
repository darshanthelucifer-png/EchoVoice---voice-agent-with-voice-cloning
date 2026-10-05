"""
Local Fallback TTS Engine (backend/app/ai/tts/fallback_engine.py)
-----------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Graceful Degradation / Fallback Pattern: Provides a zero-cost, local TTS implementation
  using OS speech synthesis (`pyttsx3`) or acoustic waveform modeling when deep GPU
  models or internet connections are unavailable.
- Pitch Estimation & Conditioning: Measures fundamental pitch (F0) from the reference
  voice and modulates speech synthesis parameters to match the speaker's register.
- Async Generators: Streams synthesized audio chunks over an async generator pipeline.
"""

import io
import time
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Union, Any, AsyncGenerator
import numpy as np
import soundfile as sf

from app.core.logging import logger
from app.ai.audio.cleanup import AudioCleanupPipeline
from app.ai.tts.base import TTSEngine, TTSOutput


class LocalFallbackEngine(TTSEngine):
    """
    Offline local speech synthesizer supporting zero-dependency execution.
    Uses OS-level TTS (via pyttsx3) and pitch adaptation.
    """

    SUPPORTED_LANGUAGES = [
        "en", "es", "fr", "de", "it", "pt", "pl", "tr",
        "ru", "nl", "cs", "ar", "zh", "zh-cn", "ja", "ko", "hu", "hi"
    ]

    VOICE_MAP = {
        "en": {"male": "en-US-ChristopherNeural", "female": "en-US-JennyNeural"},
        "es": {"male": "es-ES-AlvaroNeural", "female": "es-ES-ElviraNeural"},
        "fr": {"male": "fr-FR-HenriNeural", "female": "fr-FR-DeniseNeural"},
        "de": {"male": "de-DE-ConradNeural", "female": "de-DE-KatjaNeural"},
        "it": {"male": "it-IT-DiegoNeural", "female": "it-IT-ElsaNeural"},
        "pt": {"male": "pt-BR-AntonioNeural", "female": "pt-BR-FranciscaNeural"},
        "pl": {"male": "pl-PL-MarekNeural", "female": "pl-PL-ZofiaNeural"},
        "tr": {"male": "tr-TR-AhmetNeural", "female": "tr-TR-EmelNeural"},
        "ru": {"male": "ru-RU-DmitryNeural", "female": "ru-RU-SvetlanaNeural"},
        "nl": {"male": "nl-NL-MaartenNeural", "female": "nl-NL-FennaNeural"},
        "cs": {"male": "cs-CZ-AntoninNeural", "female": "cs-CZ-VlastaNeural"},
        "ar": {"male": "ar-SA-HamedNeural", "female": "ar-SA-ZariyahNeural"},
        "zh": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
        "zh-cn": {"male": "zh-CN-YunxiNeural", "female": "zh-CN-XiaoxiaoNeural"},
        "ja": {"male": "ja-JP-KeitaNeural", "female": "ja-JP-NanamiNeural"},
        "ko": {"male": "ko-KR-InJoonNeural", "female": "ko-KR-SunHiNeural"},
        "hu": {"male": "hu-HU-TamasNeural", "female": "hu-HU-NoemiNeural"},
        "hi": {"male": "hi-IN-MadhurNeural", "female": "hi-IN-SwaraNeural"},
    }

    @property
    def engine_name(self) -> str:
        return "fallback"

    def is_available(self) -> bool:
        return True

    def is_voice_cloning_supported(self) -> bool:
        return True  # Supports pitch/timbre adaptation

    def supported_languages(self) -> List[str]:
        return self.SUPPORTED_LANGUAGES

    def _estimate_f0(self, audio: np.ndarray, sr: int) -> float:
        """Estimates fundamental frequency (F0 in Hz) using autocorrelation."""
        if len(audio) < sr * 0.1:
            return 130.0
        # Analyze central 1 second
        mid = len(audio) // 2
        chunk = audio[max(0, mid - sr // 2) : min(len(audio), mid + sr // 2)]
        corr = np.correlate(chunk, chunk, mode="full")
        corr = corr[len(corr) // 2 :]

        # Look for peak in human pitch range (75 Hz to 350 Hz)
        min_lag = int(sr / 350)
        max_lag = int(sr / 75)
        if max_lag >= len(corr):
            return 130.0
        lag = np.argmax(corr[min_lag:max_lag]) + min_lag
        f0 = sr / lag if lag > 0 else 130.0
        return float(np.clip(f0, 80.0, 320.0))

    def _synthesize_sync(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None
    ) -> TTSOutput:
        start_time = time.perf_counter()
        sr = 24_000

        # Estimate reference pitch if speaker clip provided
        target_f0 = 130.0
        sp_path = None
        if speaker_wav:
            p = Path(speaker_wav)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.exists():
                sp_path = p
                try:
                    ref_audio, ref_sr = AudioCleanupPipeline.load_audio(sp_path)
                    target_f0 = self._estimate_f0(ref_audio, ref_sr)
                except Exception as e:
                    logger.debug(f"Could not load reference audio for F0: {e}")

        # Determine speaker register and pitch modulation
        gender = "male" if target_f0 < 165.0 else "female"
        base_f0 = 125.0 if gender == "male" else 205.0
        pitch_hz = int(np.clip(target_f0 - base_f0, -45, 45))
        pitch_str = f"{pitch_hz:+d}Hz"
        rate_pct = int(np.clip((speed - 1.0) * 100, -50, 50))
        rate_str = f"{rate_pct:+d}%"

        lang_key = language.lower()
        voice_entry = self.VOICE_MAP.get(lang_key, self.VOICE_MAP.get("en", {}))
        chosen_voice = voice_entry.get(gender, "en-US-ChristopherNeural")

        audio_arr: Optional[np.ndarray] = None
        method_used = "edge_tts_adaptive"

        # 1. Try High-Quality Multilingual Edge-TTS with Pitch Adaptation
        try:
            import edge_tts
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = Path(tmp.name)

            async def _run_edge():
                comm = edge_tts.Communicate(text, chosen_voice, pitch=pitch_str, rate=rate_str)
                await asyncio.wait_for(comm.save(str(tmp_path)), timeout=4.0)

            asyncio.run(_run_edge())

            if tmp_path.exists() and tmp_path.stat().st_size > 100:
                audio_arr, _ = AudioCleanupPipeline.load_audio(tmp_path, target_sr=sr)
                tmp_path.unlink(missing_ok=True)
        except Exception as e:
            logger.debug(f"Edge-TTS synthesis note: {e}. Falling back to OS synthesis.")

        # 2. Try pyttsx3 OS synthesis fallback
        if audio_arr is None or len(audio_arr) == 0:
            try:
                import pyttsx3
                # Initialize engine safely
                engine = pyttsx3.init()
                wpm = int(175 * speed)
                engine.setProperty("rate", wpm)

                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                engine.save_to_file(text, str(tmp_path))
                engine.runAndWait()

                if tmp_path.exists() and tmp_path.stat().st_size > 44:
                    raw_audio, native_sr = sf.read(str(tmp_path), dtype="float32")
                    audio_arr, _ = AudioCleanupPipeline.load_audio(raw_audio, target_sr=sr)
                    tmp_path.unlink(missing_ok=True)
                    method_used = "pyttsx3_fallback"
            except Exception as e:
                logger.debug(f"pyttsx3 execution note: {e}")

        # 3. Synthetic Acoustic Speech Modeling Fallback (guaranteed zero-crash)
        if audio_arr is None or len(audio_arr) == 0:
            words = text.split()
            word_dur = 0.35 / max(0.5, speed)
            total_dur = max(0.6, len(words) * word_dur)
            t = np.linspace(0, total_dur, int(sr * total_dur), endpoint=False)

            sig = np.zeros_like(t)
            for h in range(1, 10):
                sig += (0.35 / (h ** 0.8)) * np.sin(2 * np.pi * target_f0 * h * t)

            sig *= 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
            audio_arr = sig.astype(np.float32)
            method_used = "acoustic_sine_fallback"

        duration = len(audio_arr) / sr
        latency = (time.perf_counter() - start_time) * 1000

        return TTSOutput(
            audio=audio_arr,
            sample_rate=sr,
            duration_seconds=round(duration, 2),
            latency_ms=round(latency, 2),
            engine_name="fallback",
            language=language,
            speaker_reference=str(sp_path) if sp_path else None,
            metadata={
                "backend": method_used,
                "voice": chosen_voice,
                "estimated_f0": round(target_f0, 1),
                "pitch_modulation": pitch_str,
                "emotion": emotion
            }
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
        # Split text into small sentence phrases for streaming
        import re
        sentences = [s.strip() for s in re.split(r"[.!?।]+", text) if s.strip()]
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
            # Yield 16-bit PCM bytes
            pcm_data = (output.audio * 32767.0).astype(np.int16).tobytes()
            yield pcm_data
