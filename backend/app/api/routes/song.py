"""
Song Studio API Router (backend/app/api/routes/song.py)
--------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Full-Lifecycle REST API Architecture: Handles ingestion, multi-stem separation, singing
  voice conversion (SVC), multitrack remixing, broadcast mastering, and quality gate inspection.
- Asynchronous Job Polling & Telemetry: Exposes endpoints for tracking background conversions
  with stage names, progress percentages (0-100%), and ETA estimations.
- Fine-Grained Pydantic V2 Request Schemas: Validates pitch shift boundaries (-12 to +12 semitones),
  auto-tune strengths, and legal consent assertions.
- Secure Audio File Streaming: Serves separated stems, mastered tracks, and acapellas with
  proper MIME types (audio/wav, audio/mpeg, audio/flac) and path traversal protections.
"""

from pathlib import Path
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status, Query, BackgroundTasks
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import logger
from app.schemas.common import APIResponse
from app.services.song_service import (
    song_studio_service,
    SongStudioProject,
    ClonedSongRemasterResult,
    SongJobState
)
from app.ai.song.experimental import experimental_song_features, LyricTranslationResult, ACEStepGenerationResult

router = APIRouter(prefix="/song", tags=["Song Studio"])


# --- Pydantic Request Models ---

class ConvertSongRequest(BaseModel):
    song_id: str = Field(..., description="Target Song Studio project ID")
    voice_profile_id: Optional[str] = Field(default=None, description="Voice profile ID with trained RVC or enrolled reference")
    pitch_shift_semitones: float = Field(default=0.0, ge=-12.0, le=12.0, description="Pitch transpose (-12 to +12 semitones)")
    auto_tune: bool = Field(default=False, description="Enable scale-quantized pitch auto-tuning")
    autotune_strength: float = Field(default=0.70, ge=0.0, le=1.0, description="Auto-tune strength (0.0 to 1.0)")
    sidechain_duck_db: float = Field(default=2.0, ge=0.0, le=6.0, description="Pocket ducking for instrumental mids")
    force_tier: Optional[str] = Field(default=None, description="'tier3', 'tier2', or None for auto")
    user_consent: bool = Field(default=True, description="Confirmation of personal listening rights")
    voice_profile_consent: bool = Field(default=True, description="Confirmation of biometric voice cloning consent")
    async_job: bool = Field(default=False, description="Run in background and poll via job_id")


class LyricTranslationRequest(BaseModel):
    song_id: str = Field(..., description="Target Song Studio project ID")
    source_language: str = Field(default="en", description="Source vocal language code (e.g. 'en')")
    target_language: str = Field(default="hi", description="Target translation language code (e.g. 'hi', 'es', 'kn')")


class ACEStepGenerateRequest(BaseModel):
    lyrics: str = Field(..., description="Lyrics prompt for music generation")
    style_genre: str = Field(default="synthwave", description="Musical style or genre prompt")
    target_bpm: float = Field(default=120.0, ge=60.0, le=200.0, description="Target tempo BPM")


# --- Endpoints ---

@router.post(
    "/upload-and-process",
    response_model=APIResponse[Dict[str, Any]],
    summary="Upload full song & execute separation, dereverberation, and musical analysis"
)
async def upload_and_process_song(
    file: UploadFile = File(..., description="Audio file (WAV, MP3, FLAC, M4A)"),
    dereverb_strength: float = Form(default=0.50, ge=0.0, le=1.0),
    fast_mode: bool = Form(default=False),
    force_dsp: bool = Form(default=False)
):
    """
    Ingests an uploaded song, separates stems (vocals, drums, bass, other, instrumental),
    dereverberates the lead vocal track, and extracts musical key, scale, BPM, and F0 pitch contour.
    """
    allowed_extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ".wav"
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported format '{file_ext}'. Allowed formats: {allowed_extensions}"
        )

    temp_dir = settings.UPLOADS_DIR / "song_intake"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"intake_{file.filename}"

    with open(temp_path, "wb") as f_out:
        while chunk := await file.read(1024 * 1024):
            f_out.write(chunk)

    try:
        project: SongStudioProject = await song_studio_service.process_song(
            input_audio=temp_path,
            original_filename=file.filename or "song.wav",
            dereverb_strength=dereverb_strength,
            fast_mode=fast_mode,
            force_dsp=force_dsp
        )

        return APIResponse(
            success=True,
            message="Song stems separated, dereverberated, and analyzed successfully",
            data={
                "song_id": project.song_id,
                "original_filename": project.original_filename,
                "duration_seconds": project.duration_seconds,
                "sample_rate": project.sample_rate,
                "bpm": project.bpm,
                "musical_key": project.musical_key,
                "scale_type": project.scale_type,
                "scale_notes": project.scale_notes,
                "vocal_register": project.vocal_register,
                "reverb_attenuation_db": project.reverb_attenuation_db,
                "download_urls": project.download_urls,
                "f0_statistics": project.f0_stats,
                "telemetry": project.telemetry
            }
        )
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post(
    "/convert",
    response_model=APIResponse[Dict[str, Any]],
    summary="Execute singing voice conversion (SVC) and YouTube broadcast remastering"
)
async def convert_song(req: ConvertSongRequest):
    """
    Transforms the isolated lead vocal track into the user's authentic timbre,
    applies vocal post-processing (de-click, de-ess, EQ-match, timing align),
    remixes multitracks with sidechain ducking, and masters to YouTube -14 LUFS standard.
    """
    try:
        if req.async_job:
            # Start background job
            job_id = song_studio_service.start_conversion_job(
                song_id=req.song_id,
                voice_profile_id=req.voice_profile_id,
                pitch_shift_semitones=req.pitch_shift_semitones,
                auto_tune=req.auto_tune,
                autotune_strength=req.autotune_strength,
                sidechain_duck_db=req.sidechain_duck_db,
                user_consent=req.user_consent,
                voice_profile_consent=req.voice_profile_consent,
                force_tier=req.force_tier
            )
            return APIResponse(
                success=True,
                message="Singing conversion job initiated in background",
                data={
                    "job_id": job_id,
                    "song_id": req.song_id,
                    "status": "PROCESSING",
                    "poll_url": f"/api/v1/song/jobs/{job_id}"
                }
            )

        # Synchronous execution
        result: ClonedSongRemasterResult = await song_studio_service.convert_and_remaster(
            song_id=req.song_id,
            voice_profile_id=req.voice_profile_id,
            pitch_shift_semitones=req.pitch_shift_semitones,
            auto_tune=req.auto_tune,
            autotune_strength=req.autotune_strength,
            sidechain_duck_db=req.sidechain_duck_db,
            user_consent=req.user_consent,
            voice_profile_consent=req.voice_profile_consent,
            force_tier=req.force_tier
        )

        return APIResponse(
            success=True,
            message="Singing voice converted, remixed, and broadcast remastered successfully",
            data={
                "song_id": result.song_id,
                "song_title": result.song_title,
                "duration_seconds": result.duration_seconds,
                "sample_rate": result.sample_rate,
                "tier_used": result.tier_used,
                "autotune_applied": result.autotune_applied,
                "likeness_score": result.likeness_score,
                "integrated_lufs": result.integrated_lufs,
                "true_peak_db": result.true_peak_db,
                "sidechain_attenuation_db": result.sidechain_attenuation_db,
                "quality_gate_passed": result.quality_gate_passed,
                "octave_error_count": result.octave_error_count,
                "download_urls": result.download_urls,
                "telemetry": result.telemetry,
                "waveform_comparison": result.waveform_comparison
            }
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(fnf))
    except PermissionError as pe:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(pe))
    except Exception as exc:
        logger.error(f"Song conversion error: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get(
    "/jobs/{job_id}",
    response_model=APIResponse[Dict[str, Any]],
    summary="Poll status and progress of an asynchronous song conversion job"
)
async def get_song_job_status(job_id: str):
    """Retrieves stage name, progress percentage (0-100%), and ETA for a song conversion task."""
    state: Optional[SongJobState] = song_studio_service.get_job_state(job_id)
    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found."
        )

    return APIResponse(
        success=True,
        message="Song job status retrieved",
        data={
            "job_id": state.job_id,
            "song_id": state.song_id,
            "status": state.status,
            "stage": state.stage,
            "progress_percent": state.progress_percent,
            "elapsed_seconds": state.elapsed_seconds,
            "eta_seconds": state.eta_seconds,
            "result": state.result,
            "error": state.error
        }
    )


@router.post(
    "/jobs/{job_id}/cancel",
    response_model=APIResponse[Dict[str, Any]],
    summary="Cancel an active singing voice conversion job"
)
async def cancel_song_job(job_id: str):
    """Halts an active background song conversion job."""
    success = song_studio_service.cancel_job(job_id)
    return APIResponse(
        success=success,
        message="Job cancellation requested" if success else "Job was not running or not found",
        data={"job_id": job_id, "cancelled": success}
    )


@router.get(
    "/{song_id}/quality-report",
    response_model=APIResponse[Dict[str, Any]],
    summary="Retrieve acoustic, musical, and biometric quality gate audit report"
)
async def get_song_quality_report(song_id: str):
    """Returns biometric likeness scores, clipping, loudness, and flagged octave jump segments."""
    report = await song_studio_service.get_quality_report(song_id)
    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Quality report not yet generated for song '{song_id}'. Run /convert first."
        )

    return APIResponse(
        success=True,
        message="Song Quality Gate report retrieved",
        data=report
    )


@router.get(
    "/{song_id}/ab-comparison",
    response_model=APIResponse[Dict[str, Any]],
    summary="Retrieve side-by-side A/B player comparison telemetry"
)
async def get_song_ab_comparison(song_id: str):
    """Delivers original vs cloned waveforms, pitch correlations, and audio stream links for A/B testing."""
    ab_data = await song_studio_service.get_ab_comparison(song_id)
    if not ab_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"A/B comparison data not available for song '{song_id}'."
        )

    return APIResponse(
        success=True,
        message="A/B comparison data retrieved",
        data=ab_data
    )


@router.post(
    "/experimental/translate-lyrics",
    response_model=APIResponse[Dict[str, Any]],
    summary="[Experimental] Transcribe vocals with Whisper and translate lyrics via NLLB"
)
async def translate_song_lyrics(req: LyricTranslationRequest):
    """Experimental multilingual lyric translation pipeline."""
    project = await song_studio_service.get_project(req.song_id)
    if not project:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Song '{req.song_id}' not found.")

    clean_lead = project.stem_paths.get("clean_lead_vocals")
    if not clean_lead or not clean_lead.exists():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Lead vocal stem missing.")

    res: LyricTranslationResult = await experimental_song_features.translate_song_lyrics(
        vocal_audio_path=clean_lead,
        source_language=req.source_language,
        target_language=req.target_language
    )

    return APIResponse(
        success=True,
        message="Lyric translation completed (Experimental)",
        data={
            "original_transcript": res.original_transcript,
            "target_language": res.target_language,
            "translated_lyrics": res.translated_lyrics,
            "disclaimer": res.disclaimer,
            "latency_ms": res.latency_ms
        }
    )


@router.post(
    "/experimental/generate-song",
    response_model=APIResponse[Dict[str, Any]],
    summary="[Experimental] Compose new song from lyrics using ACE-Step"
)
async def generate_song_from_lyrics(req: ACEStepGenerateRequest):
    """Experimental generative music pipeline powered by ACE-Step open model architecture."""
    out_dir = settings.SONG_STUDIO_DIR / f"ace_{uuid.uuid4().hex[:6]}"
    res: ACEStepGenerationResult = await experimental_song_features.generate_song_from_lyrics(
        lyrics=req.lyrics,
        style_genre=req.style_genre,
        target_bpm=req.target_bpm,
        output_dir=out_dir
    )

    return APIResponse(
        success=True,
        message="ACE-Step generative music preview initialized (Experimental)",
        data={
            "title": res.title,
            "lyrics_prompt": res.lyrics_prompt,
            "style_genre": res.style_genre,
            "bpm": res.bpm,
            "disclaimer": res.disclaimer,
            "latency_ms": res.latency_ms
        }
    )


@router.get(
    "/{song_id}",
    response_model=APIResponse[Dict[str, Any]],
    summary="Get Song Studio project manifest and stem URLs"
)
async def get_song_project(song_id: str):
    """Retrieves metadata and stem URLs for an existing Song Studio project."""
    project = await song_studio_service.get_project(song_id)
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Song Studio project '{song_id}' not found."
        )

    return APIResponse(
        success=True,
        message="Song Studio project retrieved",
        data={
            "song_id": project.song_id,
            "original_filename": project.original_filename,
            "duration_seconds": project.duration_seconds,
            "sample_rate": project.sample_rate,
            "bpm": project.bpm,
            "musical_key": project.musical_key,
            "scale_type": project.scale_type,
            "scale_notes": project.scale_notes,
            "vocal_register": project.vocal_register,
            "reverb_attenuation_db": project.reverb_attenuation_db,
            "download_urls": project.download_urls,
            "f0_statistics": project.f0_stats,
            "telemetry": project.telemetry
        }
    )


@router.get(
    "/audio/{song_id}/{filename}",
    summary="Stream individual separated stem, acapella, or mastered audio"
)
async def stream_stem_audio(song_id: str, filename: str):
    """Delivers isolated stem audio or full masters with proper audio MIME headers."""
    safe_filename = Path(filename).name
    file_path = settings.SONG_STUDIO_DIR / song_id / safe_filename

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Audio file '{safe_filename}' not found for song '{song_id}'."
        )

    # Determine media type from extension
    ext = file_path.suffix.lower()
    media_map = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4"
    }
    media_type = media_map.get(ext, "audio/wav")

    return FileResponse(
        path=file_path,
        media_type=media_type,
        filename=safe_filename
    )
