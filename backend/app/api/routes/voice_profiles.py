"""
Voice Profiles & Studio API Router (backend/app/api/routes/voice_profiles.py)
-----------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Mandatory Consent Gate Verification: Rejects enrollment if legal consent is missing.
- Multi-part Form Processing: Coordinates audio file upload with profile metadata.
- Secure Audio File Streaming: Delivers both raw and cleaned reference clips for A/B playback.
- User Isolation: Guarantees users can only access or delete their own voice profiles.
"""

from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.api.deps import get_db, get_current_user
from app.models.user import User
from app.schemas.voice_profile import AudioQualityMetrics, VoiceProfileRead
from app.schemas.common import APIResponse
from app.services.audio_service import audio_service
from app.services.voice_profile_service import voice_profile_service
from app.ai.audio.cleanup import CleanupConfig

router = APIRouter(prefix="/voice-profiles", tags=["Voice Studio & Profiles"])


@router.post(
    "/analyze-quality",
    response_model=APIResponse[AudioQualityMetrics],
    summary="Analyze acoustic quality of voice recording (Step 1)"
)
async def analyze_voice_quality(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, M4A, FLAC)")
) -> APIResponse[AudioQualityMetrics]:
    """
    Evaluates microphone capture quality:
    SNR (dB), clipping ratio, background noise floor (dBFS), speech ratio, and tips.
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty."
        )

    report = await audio_service.analyze_audio(content)
    metrics = AudioQualityMetrics(
        snr_db=report.snr_db,
        clipping_ratio=report.clipping_ratio,
        speech_duration_seconds=report.speech_duration_seconds,
        speech_ratio=report.speech_ratio,
        noise_floor_db=report.noise_floor_db,
        is_usable=report.is_usable,
        recommendation=report.recommendation
    )
    return APIResponse(
        success=True,
        message="Audio quality analyzed successfully",
        data=metrics
    )


@router.post(
    "/preview-clean",
    summary="Preview cleaned audio for A/B player (Step 2)"
)
@router.post(
    "/clean-audio",
    summary="Clean audio endpoint (Phase 2 & 4)"
)
async def preview_clean(
    file: UploadFile = File(..., description="Raw audio recording"),
    enable_denoise: bool = Form(True),
    enable_vad_trim: bool = Form(True),
    target_lufs: float = Form(-14.0)
):
    """
    Cleans audio in-memory and returns both raw and cleaned data for A/B comparison.
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty audio file.")

    safe_name = f"preview_{Path(file.filename or 'rec').stem}.wav"
    cfg = CleanupConfig(
        enable_denoise=enable_denoise,
        enable_vad_trim=enable_vad_trim,
        target_lufs=target_lufs
    )

    out_path, result = await audio_service.clean_voice_sample(
        input_source=content,
        output_filename=safe_name,
        config=cfg
    )

    return APIResponse(
        success=True,
        message="Audio preview processed",
        data={
            "cleaned_audio_url": f"/api/v1/voice-profiles/audio/{safe_name}",
            "download_url": f"/api/v1/voice-profiles/audio/{safe_name}",
            "summary": result.summary(),
            "raw_metrics": result.raw_report.to_dict(),
            "cleaned_metrics": result.cleaned_report.to_dict(),
        }
    )


@router.post(
    "/enroll",
    response_model=APIResponse[VoiceProfileRead],
    status_code=status.HTTP_201_CREATED,
    summary="Enroll and save new voice profile with consent gate (Step 3)"
)
async def enroll_voice_profile(
    file: UploadFile = File(..., description="Voice sample (20s - 5min)"),
    name: str = Form(..., description="Name for this voice profile"),
    description: Optional[str] = Form(None),
    consent_given: bool = Form(..., description="Ethical consent checkbox confirmation"),
    enable_denoise: bool = Form(True),
    enable_vad_trim: bool = Form(True),
    target_lufs: float = Form(-14.0),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse[VoiceProfileRead]:
    """
    Consent-gated voice enrollment:
    1. Validates ethical permission
    2. Runs audio restoration
    3. Extracts speaker embedding and top 3 reference clips
    4. Persists profile in user's library
    """
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded audio file is empty."
        )

    cfg = CleanupConfig(
        enable_denoise=enable_denoise,
        enable_vad_trim=enable_vad_trim,
        target_lufs=target_lufs
    )

    profile, result = await voice_profile_service.enroll_profile(
        db=db,
        user=current_user,
        audio_bytes=content,
        name=name,
        description=description,
        consent_given=consent_given,
        cleanup_cfg=cfg
    )

    return APIResponse(
        success=True,
        message="Voice profile enrolled and created successfully",
        data=VoiceProfileRead.model_validate(profile)
    )


@router.get(
    "/",
    response_model=APIResponse[List[VoiceProfileRead]],
    summary="List all voice profiles for authenticated user"
)
async def list_user_profiles(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse[List[VoiceProfileRead]]:
    """Retrieves all voice profiles belonging to the logged-in user."""
    profiles = await voice_profile_service.get_user_profiles(db, current_user.id)
    return APIResponse(
        success=True,
        data=[VoiceProfileRead.model_validate(p) for p in profiles]
    )


@router.get(
    "/{profile_id}",
    response_model=APIResponse[VoiceProfileRead],
    summary="Get voice profile by ID"
)
async def get_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse[VoiceProfileRead]:
    """Retrieves single profile ensuring ownership."""
    profile = await voice_profile_service.get_profile_by_id(db, profile_id, current_user.id)
    if not profile:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Voice profile not found."
        )
    return APIResponse(
        success=True,
        data=VoiceProfileRead.model_validate(profile)
    )


@router.post(
    "/{profile_id}/default",
    response_model=APIResponse[VoiceProfileRead],
    summary="Set profile as user's default voice"
)
async def set_default_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
) -> APIResponse[VoiceProfileRead]:
    """Sets the chosen profile as active default for assistant replies."""
    profile = await voice_profile_service.set_default_profile(db, profile_id, current_user.id)
    return APIResponse(
        success=True,
        message=f"'{profile.name}' is now your default voice profile",
        data=VoiceProfileRead.model_validate(profile)
    )


@router.delete(
    "/{profile_id}",
    summary="Delete voice profile and securely wipe audio files"
)
async def delete_profile(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Deletes profile and completely wipes all audio clips and embeddings from disk."""
    await voice_profile_service.delete_profile(db, profile_id, current_user.id)
    return APIResponse(
        success=True,
        message="Voice profile and associated biometric audio files wiped successfully."
    )


@router.get(
    "/{profile_id}/reference-audio",
    summary="Stream clean reference audio for voice profile"
)
async def get_profile_reference_audio(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Streams the cleaned reference audio clip for browser playback."""
    profile = await voice_profile_service.get_profile_by_id(db, profile_id, current_user.id)
    if not profile or not profile.reference_audio_path or not Path(profile.reference_audio_path).exists():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    return FileResponse(
        path=profile.reference_audio_path,
        media_type="audio/wav",
        filename=f"{profile.name}_reference.wav"
    )


@router.get(
    "/{profile_id}/raw-audio",
    summary="Stream raw uncleaned audio for A/B comparison"
)
async def get_profile_raw_audio(
    profile_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Streams the original raw recording for A/B comparison."""
    profile = await voice_profile_service.get_profile_by_id(db, profile_id, current_user.id)
    if not profile or not profile.raw_audio_path or not Path(profile.raw_audio_path).exists():
        raise HTTPException(status_code=404, detail="Raw audio file not found.")
    return FileResponse(
        path=profile.raw_audio_path,
        media_type="audio/wav",
        filename=f"{profile.name}_raw.wav"
    )


@router.get(
    "/audio/{filename}",
    summary="Download preview audio clip"
)
async def get_preview_audio_file(filename: str):
    file_path = settings.VOICE_PROFILES_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)
