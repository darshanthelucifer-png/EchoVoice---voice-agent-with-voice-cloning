"""
TTS Jobs API Router (backend/app/api/routes/tts_jobs.py)
---------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- RESTful Job Resource Endpoints: Implements standard lifecycle controllers
  (Create, Read, List, Resume, Cancel, Download).
- Streaming File Responses: `FileResponse` delivers mastered broadcast audio
  (24-bit 48kHz WAV, 320kbps MP3, FLAC) with proper MIME types and attachment headers.
- Dependency Injection: Secures routes with JWT `get_current_user` and database `get_db`.
"""

from typing import List, Optional
from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user, oauth2_scheme_optional, get_current_user_flexible
from app.models.user import User
from app.schemas.tts_job import TTSJobCreate, TTSJobRead
from app.schemas.common import APIResponse
from app.services.tts_job_service import tts_job_service

router = APIRouter(prefix="/tts-jobs", tags=["Long-Form Speech Generation"])


@router.post(
    "",
    response_model=APIResponse[TTSJobRead],
    status_code=status.HTTP_201_CREATED,
    summary="Submit a new long-form speech generation job"
)
async def create_tts_job(
    payload: TTSJobCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Submits a long script, speech, or audiobook text for chunked synthesis,
    resilient checkpointing, and YouTube broadcast mastering.
    """
    job = await tts_job_service.create_job(db, current_user, payload)
    return APIResponse(
        data=TTSJobRead.model_validate(job),
        message="Long-form speech generation job scheduled successfully."
    )


@router.get(
    "",
    response_model=APIResponse[List[TTSJobRead]],
    summary="List all speech generation jobs for current user"
)
async def list_tts_jobs(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieves all past and active generation jobs."""
    jobs = await tts_job_service.list_jobs(db, current_user.id, limit=limit, offset=offset)
    return APIResponse(
        data=[TTSJobRead.model_validate(j) for j in jobs]
    )


@router.get(
    "/{job_id}",
    response_model=APIResponse[TTSJobRead],
    summary="Get status, progress, and QC telemetry of a specific job"
)
async def get_tts_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Returns real-time synthesis progress, completed chunks, and QC report."""
    job = await tts_job_service.get_job(db, job_id, current_user.id)
    return APIResponse(data=TTSJobRead.model_validate(job))


@router.post(
    "/{job_id}/resume",
    response_model=APIResponse[TTSJobRead],
    summary="Resume an interrupted or paused generation job"
)
async def resume_tts_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Resumes processing from the last saved chunk checkpoint without repeating work."""
    job = await tts_job_service.resume_job(db, job_id, current_user.id)
    return APIResponse(
        data=TTSJobRead.model_validate(job),
        message="Job resumed from last saved checkpoint."
    )


@router.post(
    "/{job_id}/cancel",
    response_model=APIResponse[TTSJobRead],
    summary="Cancel an ongoing speech generation job"
)
async def cancel_tts_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Stops generation and preserves existing completed chunk checkpoints."""
    job = await tts_job_service.cancel_job(db, job_id, current_user.id)
    return APIResponse(
        data=TTSJobRead.model_validate(job),
        message="Job cancelled."
    )


@router.get(
    "/{job_id}/download/{format_type}",
    summary="Download mastered broadcast audio (wav, mp3, flac)"
)
async def download_mastered_audio(
    job_id: str,
    format_type: str,
    token: Optional[str] = Query(None),
    bearer_token: Optional[str] = Depends(oauth2_scheme_optional),
    db: AsyncSession = Depends(get_db)
):
    """
    Downloads or streams the YouTube-mastered audio file (-14 LUFS, -1.5 dBTP)
    in the requested format. Supports both Bearer Authorization header and ?token= query parameter.
    """
    current_user = await get_current_user_flexible(
        db=db,
        token_header=bearer_token,
        token_query=token
    )
    user_id = current_user.id if current_user else None

    file_path, media_type, download_name = await tts_job_service.get_export_file(
        db, job_id, user_id, format_type
    )
    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=download_name
    )
