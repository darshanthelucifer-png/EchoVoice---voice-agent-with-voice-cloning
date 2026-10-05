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

    VOICE_CLONE_SPACES = [
        {"space": "tonyassi/voice-clone", "endpoint": "/clone", "param": "audio"},
        {"space": "SachinAmliyar15/voice-clone-free", "endpoint": "/clone_voice", "param": "audio_sample"},
    ]

    def _is_space_running(self, space_id: str) -> bool:
        """Fast check to verify space is currently in RUNNING state on Hugging Face."""
        disabled_attr = f"_disabled_{space_id.replace('/', '_')}"
        if getattr(self, disabled_attr, False):
            disabled_time = getattr(self, f"{disabled_attr}_time", 0)
            if time.time() - disabled_time < 180:
                return False
        try:
            import urllib.request
            import json
            req = urllib.request.Request(
                f"https://huggingface.co/api/spaces/{space_id}",
                headers={"User-Agent": "EchoVoice-TTS/1.0"}
            )
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                stage = data.get("runtime", {}).get("stage")
                if stage == "RUNNING":
                    return True
                setattr(self, disabled_attr, True)
                setattr(self, f"{disabled_attr}_time", time.time())
                return False
        except Exception:
            setattr(self, disabled_attr, True)
            setattr(self, f"{disabled_attr}_time", time.time())
            return False

    def _synthesize_via_hf_space(
        self,
        text: str,
        speaker_wav: Union[str, Path],
        language: str,
        speed: float = 1.0
    ) -> Optional[np.ndarray]:
        """Synthesize zero-shot voice clone using active Hugging Face GPU Spaces via gradio_client."""
        if getattr(self, "_zerogpu_quota_exceeded", False):
            if time.time() - getattr(self, "_zerogpu_quota_time", 0) < 1800:
                return None

        from gradio_client import Client, handle_file

        sp_path = Path(speaker_wav)
        if not sp_path.is_absolute():
            sp_path = (Path.cwd() / sp_path).resolve()
        if not sp_path.exists():
            logger.warning(f"Speaker reference audio not found at: {sp_path}")
            return None

        # Check configured custom space or fallback to our active GPU spaces
        spaces_to_try = []
        if self._hf_space and self._hf_space not in ["coqui/xtts"]:
            spaces_to_try.append({"space": self._hf_space, "endpoint": "/clone", "param": "audio"})
        spaces_to_try.extend(self.VOICE_CLONE_SPACES)

        for s_cfg in spaces_to_try:
            space_id = s_cfg["space"]
            endpoint = s_cfg["endpoint"]
            audio_param = s_cfg["param"]

            if not self._is_space_running(space_id):
                continue

            try:
                logger.info(f"Invoking neural voice cloning on GPU Space '{space_id}' ({endpoint})...")
                token = settings.HF_TOKEN if settings.HF_TOKEN else None
                client = Client(space_id, token=token)

                kwargs = {
                    "text": text,
                    audio_param: handle_file(str(sp_path)),
                    "api_name": endpoint
                }
                if endpoint == "/clone_voice":
                    kwargs["language"] = "hi" if language.lower() in ["hi", "hindi"] else "en"

                result = client.predict(**kwargs)

                if result and Path(result).exists():
                    audio, _ = AudioCleanupPipeline.load_audio(result, target_sr=24000)
                    logger.info(f"Voice cloning successful from '{space_id}' ({len(audio)/24000:.2f}s audio generated)")
                    return audio
            except Exception as e:
                err_str = str(e)
                logger.warning(f"Voice cloning space '{space_id}' attempt note: {err_str}")
                disabled_attr = f"_disabled_{space_id.replace('/', '_')}"
                setattr(self, disabled_attr, True)
                setattr(self, f"{disabled_attr}_time", time.time())
                if "ZeroGPU runs limit" in err_str:
                    logger.info("ZeroGPU free quota limit active. Seamlessly using local pitch-adaptive neural voice engine.")
                    self._zerogpu_quota_exceeded = True
                    self._zerogpu_quota_time = time.time()
                    break

        return None

    @timed_step("XTTS-v2 Voice Synthesis")
    def _synthesize_sync(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        temperature: float = 0.25,
        fixed_seed: Optional[int] = 42,
        **kwargs: Any
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

        # Resolve speaker reference: check if given path or nearby clips/clip_1.wav exists
        resolved_speaker: Optional[Path] = None
        if speaker_wav:
            p = Path(speaker_wav)
            if not p.is_absolute():
                p = (Path.cwd() / p).resolve()
            if p.is_dir():
                best_clip = p / "clips" / "clip_1.wav"
                ref_wav = p / "reference.wav"
                if best_clip.exists():
                    resolved_speaker = best_clip
                elif ref_wav.exists():
                    resolved_speaker = ref_wav
            elif p.exists():
                resolved_speaker = p

        # 1. Attempt Local Coqui XTTS if speaker wav is provided
        if resolved_speaker:
            audio_arr = self._synthesize_via_local_coqui(text, resolved_speaker, lang, speed)
            if audio_arr is not None:
                method_used = "local_coqui"

        # 2. Attempt Remote Hugging Face Space
        if audio_arr is None and resolved_speaker:
            audio_arr = self._synthesize_via_hf_space(text, resolved_speaker, lang, speed)
            if audio_arr is not None:
                method_used = "hf_space_gradio"

        # 3. Graceful Fallback
        if audio_arr is None:
            fallback_out = self._fallback._synthesize_sync(
                text=text,
                speaker_wav=resolved_speaker or speaker_wav,
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
            speaker_reference=str(resolved_speaker or speaker_wav) if (resolved_speaker or speaker_wav) else None,
            metadata={
                "backend": method_used,
                "tier": "Tier 1: Zero-shot XTTS-v2",
                "emotion": emotion,
                "speed": speed,
                "temperature": temperature,
                "fixed_seed": fixed_seed
            }
        )

    async def synthesize(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path]] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        temperature: float = 0.25,
        fixed_seed: Optional[int] = 42,
        **kwargs: Any
    ) -> TTSOutput:
        return await asyncio.to_thread(
            self._synthesize_sync,
            text,
            speaker_wav,
            language,
            speed,
            emotion,
            temperature,
            fixed_seed,
            **kwargs
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
