"""
TTS & Voice Cloning Router (backend/app/api/routes/tts.py)
----------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- RESTful Audio Synthesis & Streaming Endpoints: Translates JSON requests into
  audio files or streams.
- A/B Perceptual Benchmark Endpoint: Exposes side-by-side engine and voice comparison
  for user evaluation.
- OpenAPI Tags & Documentation: Detailed parameter schemas for Swagger UI.
"""

from typing import Optional
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.core.logging import logger
from app.schemas.common import APIResponse
from app.services.tts_service import tts_service
from app.ai.tts.registry import tts_registry

router = APIRouter(prefix="/tts", tags=["TTS & Voice Cloning"])


class SynthesisRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000, description="Short text to synthesize")
    language: str = Field(default="en", description="Target ISO language code (e.g. 'en', 'es', 'hi')")
    speaker_wav_path: Optional[str] = Field(None, description="Path to clean reference audio for voice cloning")
    voice_profile_id: Optional[str] = Field(None, description="ID of enrolled voice profile in user library")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="Playback speed factor")
    emotion: Optional[str] = Field(None, description="Emotion style hint ('calm', 'warm', 'excited', 'serious')")
    engine: Optional[str] = Field(None, description="Override default TTS engine ('xtts_v2', 'fallback')")
    auto_tier_selection: bool = Field(default=True, description="Auto-escalate to Tier 2 Seed-VC if likeness < 0.85")
    target_likeness: float = Field(default=0.85, ge=0.5, le=1.0, description="Minimum likeness score threshold")
    force_tier: Optional[str] = Field(None, description="Force specific tier ('tier1', 'tier2', 'tier3')")


class ABTestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Text script to compare across options")
    engine_a: str = Field(default="xtts_v2", description="Engine for Option A")
    engine_b: str = Field(default="fallback", description="Engine for Option B")
    speaker_wav_path: Optional[str] = Field(None, description="Reference audio clip to clone")
    language: str = Field(default="en", description="Target language")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)


@router.get("/engines", summary="List registered TTS engines and capabilities")
async def list_engines():
    """Returns all registered TTS engines in the system."""
    engines = tts_registry.available_engines()
    return APIResponse(
        success=True,
        data={
            "active_engine": settings.TTS_ENGINE,
            "registered_engines": engines,
        }
    )


@router.post("/synthesize", summary="Synthesize short text with voice cloning & multi-tier auto-selection")
async def synthesize_text(req: SynthesisRequest):
    """
    Synthesizes short text into spoken audio conditioned on the speaker reference clip
    or enrolled voice profile. Automatically triggers Tier 2 Seed-VC if Tier 1 likeness < 0.85.
    """
    try:
        output, save_path = await tts_service.generate_speech(
            text=req.text,
            speaker_wav=req.speaker_wav_path,
            voice_profile_id=req.voice_profile_id,
            language=req.language,
            speed=req.speed,
            emotion=req.emotion,
            engine_name=req.engine,
            auto_tier_selection=req.auto_tier_selection,
            target_likeness=req.target_likeness,
            force_tier=req.force_tier
        )

        return APIResponse(
            success=True,
            message="Speech synthesized successfully",
            data={
                "download_url": f"/api/v1/tts/audio/{save_path.name}",
                "filename": save_path.name,
                "duration_seconds": output.duration_seconds,
                "latency_ms": output.latency_ms,
                "rtf": round(output.rtf, 3),
                "engine": output.engine_name,
                "language": output.language,
                "metadata": output.metadata
            }
        )
    except Exception as exc:
        logger.error(f"Synthesis failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech synthesis error: {str(exc)}"
        )


@router.post("/ab-test", summary="Run side-by-side A/B comparison test (Phase 3)")
async def run_ab_test_endpoint(req: ABTestRequest):
    """
    Synthesizes the exact same text across two engines or reference voices
    to evaluate timbre and latency.
    """
    try:
        results = await tts_service.run_ab_test(
            text=req.text,
            speaker_wav_a=req.speaker_wav_path,
            engine_name_a=req.engine_a,
            engine_name_b=req.engine_b,
            language=req.language,
            speed=req.speed
        )
        return APIResponse(
            success=True,
            message="A/B test completed successfully",
            data=results
        )
    except Exception as exc:
        logger.error(f"A/B test failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"A/B test execution error: {str(exc)}"
        )


@router.get("/audio/{filename}", summary="Download or stream synthesized speech")
async def get_synthesized_audio(filename: str):
    """Streams the synthesized WAV file for playback in the web player."""
    file_path = settings.EXPORTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{filename}' not found."
        )
    return FileResponse(
        path=str(file_path),
        media_type="audio/wav",
        filename=filename
    )


# =========================================================================
# PHASE 4: END-TO-END PIPELINE & BLIND A/B TEST ENDPOINTS
# =========================================================================

class ExactnessSettingsSchema(BaseModel):
    temperature: float = Field(default=0.15, ge=0.0, le=1.0, description="Sampling temperature")
    seed: Optional[int] = Field(default=42, description="Fixed seed for deterministic reproduction")
    pitch_shift: float = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch shift in semitones")
    index_rate: float = Field(default=0.75, ge=0.0, le=1.0, description="Faiss feature retrieval influence")
    protect_consonants: float = Field(default=0.33, ge=0.0, le=1.0, description="Consonant protection factor")
    accent_mode: str = Field(default="keep_accent", description="'keep_accent' vs 'native_accent'")
    mastering_enabled: bool = Field(default=True, description="Enable YouTube broadcast mastering")
    target_lufs: float = Field(default=-14.0, description="Integrated loudness target (-14 LUFS)")


class EndToEndPipelineRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Text to synthesize")
    language: str = Field(default="en", description="Target language")
    speaker_wav_path: Optional[str] = Field(None, description="Path to clean speaker audio")
    voice_profile_id: Optional[str] = Field(None, description="Voice profile ID")
    speed: float = Field(default=1.0, ge=0.5, le=2.0)
    emotion: Optional[str] = Field(None, description="Emotion style")
    base_engine: str = Field(default="fallback", description="Base TTS engine")
    exactness: Optional[ExactnessSettingsSchema] = None


class BlindABTestRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=500, description="Text script for blind testing")
    speaker_wav_path: Optional[str] = None
    voice_profile_id: Optional[str] = None
    config_a: Optional[dict] = Field(default=None, description="Overrides for Option A (e.g. {'force_tier': 'tier1', 'mastering_enabled': False})")
    config_b: Optional[dict] = Field(default=None, description="Overrides for Option B (e.g. {'force_tier': 'tier3', 'mastering_enabled': True})")
    language: str = "en"


class RevealBlindTestRequest(BaseModel):
    test_id: str = Field(..., description="Blind test session ID")
    user_vote: Optional[str] = Field(None, description="User preference ('Candidate 1', 'Candidate 2', or 'Undecided')")


class AutoJudgeRequest(BaseModel):
    test_sentence: str = Field(..., min_length=1, max_length=500, description="Calibration sentence to test across all tiers")
    speaker_wav_path: Optional[str] = None
    voice_profile_id: Optional[str] = None
    language: str = "en"


@router.post("/pipeline", summary="End-to-End Speech Pipeline (Text -> TTS -> VC -> Enhance -> Master)")
async def run_end_to_end_pipeline(req: EndToEndPipelineRequest):
    """
    Executes the complete 5-stage speech synthesis pipeline:
    1. Base TTS Synthesis (Pronunciation & lexical timing)
    2. Voice Conversion to User Voice (Tier 1 / Tier 2 / Tier 3 RVC)
    3. Output-Side Cleanup & Enhancement (De-essing, HPF 80Hz, VAD Pause Capping)
    4. YouTube-Standard Mastering (Polyphase 48kHz, Voiceover EQ, Leveler, -14 LUFS, -1.5 dBTP)
    5. Biometric Verification & Quality Telemetry (ECAPA-TDNN & WavLM-SV)
    """
    from app.services.speech_pipeline_service import speech_pipeline_service, PipelineExactnessSettings
    try:
        exact = PipelineExactnessSettings(
            temperature=req.exactness.temperature if req.exactness else 0.15,
            seed=req.exactness.seed if req.exactness else 42,
            pitch_shift=req.exactness.pitch_shift if req.exactness else 0.0,
            index_rate=req.exactness.index_rate if req.exactness else 0.75,
            protect_consonants=req.exactness.protect_consonants if req.exactness else 0.33,
            accent_mode=req.exactness.accent_mode if req.exactness else "keep_accent",
            mastering_enabled=req.exactness.mastering_enabled if req.exactness else True,
            target_lufs=req.exactness.target_lufs if req.exactness else -14.0
        )
        res = await speech_pipeline_service.run_pipeline(
            text=req.text,
            speaker_wav=req.speaker_wav_path,
            voice_profile_id=req.voice_profile_id,
            language=req.language,
            speed=req.speed,
            emotion=req.emotion,
            base_engine=req.base_engine,
            exactness=exact
        )
        return APIResponse(
            success=True,
            message="End-to-end speech pipeline executed successfully",
            data={
                "duration_seconds": res.duration_seconds,
                "sample_rate": res.sample_rate,
                "integrated_lufs": res.integrated_lufs,
                "true_peak_db": res.true_peak_db,
                "ecapa_similarity": res.ecapa_similarity,
                "wavlm_similarity": res.wavlm_similarity,
                "composite_likeness": res.composite_likeness,
                "passed_gate": res.passed_gate,
                "tier_used": res.telemetry.tier_used,
                "telemetry": {
                    "base_tts_latency_ms": res.telemetry.base_tts_latency_ms,
                    "voice_conversion_latency_ms": res.telemetry.voice_conversion_latency_ms,
                    "output_enhancement_latency_ms": res.telemetry.output_enhancement_latency_ms,
                    "mastering_latency_ms": res.telemetry.mastering_latency_ms,
                    "verification_latency_ms": res.telemetry.verification_latency_ms,
                    "total_pipeline_latency_ms": res.telemetry.total_pipeline_latency_ms,
                    "intermediate_sample_rates": res.telemetry.intermediate_sample_rates
                },
                "exactness_settings": res.exactness_settings,
                "download_urls": res.download_urls
            }
        )
    except Exception as exc:
        logger.error(f"End-to-end pipeline failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech pipeline error: {str(exc)}"
        )


@router.post("/ab-blind-test", summary="Initiate Randomized Double-Blind Perceptual Test")
async def create_blind_test_endpoint(req: BlindABTestRequest):
    """
    Synthesizes two randomized, unlabeled candidates ('Candidate 1' vs 'Candidate 2')
    with scrambled identities for unbiased listening evaluation.
    """
    from app.services.ab_testing_service import ab_testing_service
    try:
        data = await ab_testing_service.create_blind_test(
            text=req.text,
            speaker_wav=req.speaker_wav_path,
            voice_profile_id=req.voice_profile_id,
            config_a=req.config_a,
            config_b=req.config_b,
            language=req.language
        )
        return APIResponse(
            success=True,
            message="Blind A/B test generated successfully. Identities are hidden.",
            data=data
        )
    except Exception as exc:
        logger.error(f"Blind A/B test creation failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blind test error: {str(exc)}"
        )


@router.post("/ab-blind-test/reveal", summary="Reveal True Identities and Biometric Scores of Blind Test")
async def reveal_blind_test_endpoint(req: RevealBlindTestRequest):
    """
    Reveals model names, tiers, mastering status, loudness levels, and biometric scores
    for a completed blind test.
    """
    from app.services.ab_testing_service import ab_testing_service
    try:
        data = ab_testing_service.reveal_blind_test(
            test_id=req.test_id,
            user_vote=req.user_vote
        )
        return APIResponse(
            success=True,
            message="Blind test revealed successfully",
            data=data
        )
    except KeyError as k_err:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(k_err))
    except Exception as exc:
        logger.error(f"Blind test reveal failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blind test reveal error: {str(exc)}"
        )


@router.post("/auto-judge", summary="Auto-Run and Rank All Available Tiers on a Test Sentence")
async def auto_judge_all_tiers_endpoint(req: AutoJudgeRequest):
    """
    Automatically executes all available tiers (Tier 1, Tier 2, Tier 3) on a test sentence,
    evaluates biometric similarity, and picks the highest scoring engine.
    """
    from app.services.ab_testing_service import ab_testing_service
    try:
        data = await ab_testing_service.auto_judge_all_tiers(
            test_sentence=req.test_sentence,
            speaker_wav=req.speaker_wav_path,
            voice_profile_id=req.voice_profile_id,
            language=req.language
        )
        return APIResponse(
            success=True,
            message="Auto-judge evaluation completed",
            data=data
        )
    except Exception as exc:
        logger.error(f"Auto-judge failed: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Auto-judge error: {str(exc)}"
        )

