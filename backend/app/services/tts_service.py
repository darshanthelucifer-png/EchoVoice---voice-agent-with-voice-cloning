"""
TTS & Voice Cloning Service (backend/app/services/tts_service.py)
-----------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Facade Pattern: Coordinates text preprocessing, engine selection via `TTSRegistry`,
  audio post-mastering, and file saving behind a clean service API.
- A/B Testing Workflow: Synthesizes matching scripts across two distinct engines
  or reference voice clips to benchmark latency, RTF, and perceptual timbre.
- Async Thread Delegation: Ensures synthesis never starves the main ASGI event loop.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
import numpy as np

from app.core.config import settings
from app.core.logging import logger
from app.ai.audio.cleanup import AudioCleanupPipeline
from app.ai.tts.base import TTSOutput
from app.ai.tts.registry import get_tts_engine
from app.services.voice_engine_service import voice_engine_service, TieredSynthesisResult


class TTSService:
    """
    Manages speech synthesis requests, voice cloning conditioning,
    multi-tier auto-selection (Tier 1 XTTS -> Tier 2 Seed-VC -> Tier 3 RVC),
    and A/B perceptual benchmarking.
    """

    async def generate_speech(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path, np.ndarray]] = None,
        voice_profile_id: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        engine_name: Optional[str] = None,
        output_filename: Optional[str] = None,
        auto_tier_selection: bool = True,
        target_likeness: float = 0.85,
        force_tier: Optional[str] = None
    ) -> Tuple[TTSOutput, Path]:
        """
        Synthesizes speech with optional multi-tier voice cloning (Tier 1 zero-shot
        escalating to Tier 2 Seed-VC if likeness is < 0.85). Persists output WAV.
        """
        clean_text = text.strip()

        # If a voice reference or profile is provided and auto-tier or forced tier is enabled
        has_speaker_target = bool(speaker_wav or voice_profile_id)
        if has_speaker_target and (auto_tier_selection or force_tier):
            tiered_res: TieredSynthesisResult = await voice_engine_service.synthesize(
                text=clean_text,
                speaker_wav=speaker_wav,
                profile_id=voice_profile_id,
                language=language,
                speed=speed,
                emotion=emotion,
                base_engine=engine_name or settings.TTS_ENGINE,
                auto_tier_selection=auto_tier_selection,
                target_likeness=target_likeness,
                force_tier=force_tier
            )

            fname = output_filename or f"tts_{int(tiered_res.total_latency_ms)}_{tiered_res.tier_used}.wav"
            save_path = settings.EXPORTS_DIR / fname
            AudioCleanupPipeline.save_audio(tiered_res.audio, tiered_res.sample_rate, save_path)

            output = TTSOutput(
                audio=tiered_res.audio,
                sample_rate=tiered_res.sample_rate,
                duration_seconds=tiered_res.duration_seconds,
                latency_ms=tiered_res.total_latency_ms,
                engine_name=tiered_res.tier_used,
                language=language,
                speaker_reference=str(speaker_wav) if speaker_wav else voice_profile_id,
                metadata={
                    "tier_used": tiered_res.tier_used,
                    "tier1_likeness": tiered_res.tier1_likeness,
                    "tier2_likeness": tiered_res.tier2_likeness,
                    "final_likeness": tiered_res.final_likeness,
                    "passed_gate": tiered_res.passed_gate,
                    "user_flag": tiered_res.user_flag,
                    **tiered_res.telemetry
                }
            )
            return output, save_path

        # Standard direct engine synthesis
        engine = get_tts_engine(engine_name)
        output = await engine.synthesize(
            text=clean_text,
            speaker_wav=speaker_wav,
            language=language,
            speed=speed,
            emotion=emotion
        )

        fname = output_filename or f"tts_{int(output.latency_ms)}_{engine.engine_name}.wav"
        save_path = settings.EXPORTS_DIR / fname
        AudioCleanupPipeline.save_audio(output.audio, output.sample_rate, save_path)

        return output, save_path

    async def run_ab_test(
        self,
        text: str,
        speaker_wav_a: Optional[Union[str, Path]] = None,
        speaker_wav_b: Optional[Union[str, Path]] = None,
        engine_name_a: str = "xtts_v2",
        engine_name_b: str = "fallback",
        language: str = "en",
        speed: float = 1.0
    ) -> Dict[str, Any]:
        """
        Executes a side-by-side A/B comparison test on the same input script.
        Compares latency, RTF (Real-Time Factor), duration, and generated audio.
        """
        # Run Option A
        out_a, path_a = await self.generate_speech(
            text=text,
            speaker_wav=speaker_wav_a,
            language=language,
            speed=speed,
            engine_name=engine_name_a,
            output_filename="ab_test_option_A.wav"
        )

        # Run Option B
        out_b, path_b = await self.generate_speech(
            text=text,
            speaker_wav=speaker_wav_b or speaker_wav_a,
            language=language,
            speed=speed,
            engine_name=engine_name_b,
            output_filename="ab_test_option_B.wav"
        )

        return {
            "text": text,
            "language": language,
            "option_a": {
                "engine": engine_name_a,
                "speaker_reference": str(speaker_wav_a) if speaker_wav_a else "default",
                "duration_seconds": out_a.duration_seconds,
                "latency_ms": out_a.latency_ms,
                "rtf": round(out_a.rtf, 3),
                "audio_path": str(path_a),
                "download_url": f"/api/v1/tts/audio/{path_a.name}",
            },
            "option_b": {
                "engine": engine_name_b,
                "speaker_reference": str(speaker_wav_b) if speaker_wav_b else (str(speaker_wav_a) if speaker_wav_a else "default"),
                "duration_seconds": out_b.duration_seconds,
                "latency_ms": out_b.latency_ms,
                "rtf": round(out_b.rtf, 3),
                "audio_path": str(path_b),
                "download_url": f"/api/v1/tts/audio/{path_b.name}",
            },
            "latency_delta_ms": round(out_b.latency_ms - out_a.latency_ms, 2),
        }


tts_service = TTSService()
