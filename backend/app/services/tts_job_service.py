"""
TTS Long-Form Job Service (backend/app/services/tts_job_service.py)
-------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Service Layer Pattern: Encapsulates database transactions, authorization checks,
  and background task delegation away from HTTP controllers.
- Checkpointing Coordination: Coordinates job initiation, status polling, pause/resume,
  and cancellation across worker pools.
"""

from pathlib import Path
from typing import List, Optional, Tuple
from fastapi import HTTPException, status
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.tts_job import TTSJob, JobStatus
from app.models.voice_profile import VoiceProfile
from app.models.user import User
from app.schemas.tts_job import TTSJobCreate
from app.workers.job_runner import job_runner
from app.core.logging import logger, timed_step


class TTSJobService:
    """
    Manages long-form text-to-speech jobs, coordinating database state,
    voice profile resolution, and background worker checkpointing.
    """

    @timed_step("Create TTS Long-Form Job")
    async def create_job(
        self,
        db: AsyncSession,
        user: User,
        payload: TTSJobCreate
    ) -> TTSJob:
        """
        Creates a new long-form synthesis job and schedules execution on the worker pool.
        """
        # Validate voice profile ownership if specified
        if payload.voice_profile_id:
            vp_res = await db.execute(
                select(VoiceProfile).where(
                    VoiceProfile.id == payload.voice_profile_id,
                    VoiceProfile.user_id == user.id
                )
            )
            if not vp_res.scalar_one_or_none():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Specified voice profile does not exist or does not belong to you."
                )

        # Create TTSJob record
        job = TTSJob(
            user_id=user.id,
            voice_profile_id=payload.voice_profile_id,
            status=JobStatus.PENDING.value,
            script_text=payload.script_text,
            target_language=payload.target_language,
            engine=payload.engine,
            mastering_preset=payload.mastering_preset,
            progress=0.0,
            total_chunks=0,
            completed_chunks=0
        )
        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Dispatch background worker
        await job_runner.start_job(job.id)

        logger.info(f"Dispatched long-form speech job {job.id} for user {user.id}")
        return job

    async def get_job(
        self,
        db: AsyncSession,
        job_id: str,
        user_id: str
    ) -> TTSJob:
        """Retrieves job record, ensuring user ownership."""
        res = await db.execute(
            select(TTSJob).where(TTSJob.id == job_id, TTSJob.user_id == user_id)
        )
        job = res.scalar_one_or_none()
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="TTS Job not found."
            )
        return job

    async def list_jobs(
        self,
        db: AsyncSession,
        user_id: str,
        limit: int = 50,
        offset: int = 0
    ) -> List[TTSJob]:
        """Lists all synthesis jobs for the current user."""
        res = await db.execute(
            select(TTSJob)
            .where(TTSJob.user_id == user_id)
            .order_by(desc(TTSJob.created_at))
            .limit(limit)
            .offset(offset)
        )
        return list(res.scalars().all())

    async def resume_job(
        self,
        db: AsyncSession,
        job_id: str,
        user_id: str
    ) -> TTSJob:
        """Resumes an interrupted or failed long-form generation job."""
        job = await self.get_job(db, job_id, user_id)
        if job.status == JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job is already completed."
            )

        job.status = JobStatus.PROCESSING.value
        job.error_message = None
        await db.commit()
        await db.refresh(job)

        await job_runner.resume_job(job_id)
        logger.info(f"Resumed job {job_id} from last saved checkpoint.")
        return job

    async def cancel_job(
        self,
        db: AsyncSession,
        job_id: str,
        user_id: str
    ) -> TTSJob:
        """Cancels an actively running job."""
        job = await self.get_job(db, job_id, user_id)
        if job.status in (JobStatus.COMPLETED.value, JobStatus.CANCELLED.value):
            return job

        job_runner.cancel_job(job_id)
        job.status = JobStatus.CANCELLED.value
        await db.commit()
        await db.refresh(job)
        return job

    async def get_export_file(
        self,
        db: AsyncSession,
        job_id: str,
        user_id: str,
        format_type: str
    ) -> Tuple[Path, str, str]:
        """
        Retrieves file path and mime type for mastered audio exports.
        Returns: (file_path, media_type, download_filename)
        """
        job = await self.get_job(db, job_id, user_id)

        if job.status != JobStatus.COMPLETED.value:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Audio export is not ready. Current job status: {job.status}"
            )

        fmt = format_type.lower()
        if fmt in ("wav", "24bit"):
            file_path = Path(job.output_wav_path) if job.output_wav_path else None
            media_type = "audio/wav"
            dl_name = f"echovoice_{job.id[:8]}_master.wav"
        elif fmt in ("mp3", "320k"):
            file_path = Path(job.output_mp3_path) if job.output_mp3_path else None
            media_type = "audio/mpeg"
            dl_name = f"echovoice_{job.id[:8]}_master.mp3"
        elif fmt in ("flac", "m4a", "lossless"):
            file_path = Path(job.output_m4a_path) if job.output_m4a_path else None
            media_type = "audio/flac"
            dl_name = f"echovoice_{job.id[:8]}_master.flac"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported format '{format_type}'. Choose from 'wav', 'mp3', or 'flac'."
            )

        if not file_path or not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Exported {fmt.upper()} audio file could not be found on disk."
            )

        return file_path, media_type, dl_name


tts_job_service = TTSJobService()
