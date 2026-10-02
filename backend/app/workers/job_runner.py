"""
Resilient Long-Form Speech Job Runner (backend/app/workers/job_runner.py)
------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Checkpoint / Resume Pattern: Atomic JSON checkpointing on every chunk ensures long jobs
  (10-minute to 1-hour audiobooks/podcasts) survive server crashes and can resume instantly
  without restarting from chunk 0.
- State Machine Life-Cycle: PENDING -> PROCESSING -> COMPLETED / FAILED / CANCELLED.
- Background Task Coordination: Asynchronous task tracking with thread-safe cancellation
  and non-blocking execution off FastAPI's main ASGI thread.
- Multi-Stage Pipeline Orchestration: Chunker -> TTS Synthesis -> Audio Stitcher
  -> YouTube Mastering Engine -> Audio QC Evaluator.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any
import asyncio
import numpy as np
import soundfile as sf
from sqlalchemy import select, update

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.core.logging import logger, timed_step
from app.models.tts_job import TTSJob, JobStatus
from app.models.voice_profile import VoiceProfile
from app.ai.audio.chunker import chunker, ChunkItem
from app.ai.audio.stitcher import audio_stitcher
from app.ai.audio.mastering import youtube_mastering_engine
from app.ai.audio.qc import audio_qc_evaluator
from app.services.tts_service import TTSService


class LongFormJobRunner:
    """
    Manages long-form speech generation jobs with chunk-level atomic checkpointing,
    pause/resume capabilities, live progress telemetry, and YouTube mastering.
    """

    def __init__(self):
        self._active_tasks: Dict[str, asyncio.Task] = {}
        self._cancelled_jobs: Set[str] = set()
        self.tts_service = TTSService()

    def get_checkpoint_dir(self, job_id: str) -> Path:
        """Returns the isolated checkpoint directory for a job."""
        d = settings.CHECKPOINTS_DIR / job_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "chunks").mkdir(parents=True, exist_ok=True)
        return d

    def _write_checkpoint_json(self, checkpoint_path: Path, state: Dict[str, Any]) -> None:
        """Atomically writes checkpoint data via temp file swap."""
        temp_path = checkpoint_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)
        temp_path.replace(checkpoint_path)

    def _load_checkpoint_json(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Loads checkpoint state if present."""
        if not checkpoint_path.exists():
            return None
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning(f"Failed to read checkpoint at {checkpoint_path}: {exc}")
            return None

    async def start_job(self, job_id: str) -> asyncio.Task:
        """Schedules job execution in the background."""
        if job_id in self._active_tasks and not self._active_tasks[job_id].done():
            logger.warning(f"Job {job_id} is already actively running.")
            return self._active_tasks[job_id]

        self._cancelled_jobs.discard(job_id)
        task = asyncio.create_task(self._execute_job(job_id))
        self._active_tasks[job_id] = task
        return task

    async def resume_job(self, job_id: str) -> asyncio.Task:
        """Resumes an interrupted or paused long-form generation job."""
        return await self.start_job(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Signals active job to halt gracefully."""
        self._cancelled_jobs.add(job_id)
        task = self._active_tasks.get(job_id)
        if task and not task.done():
            task.cancel()
            logger.info(f"Cancellation requested for job {job_id}")
            return True
        return False

    async def _execute_job(self, job_id: str) -> None:
        """Main execution loop for a long-form speech synthesis job."""
        logger.info(f"Starting long-form job execution: {job_id}")
        checkpoint_dir = self.get_checkpoint_dir(job_id)
        checkpoint_file = checkpoint_dir / "checkpoint.json"
        chunks_dir = checkpoint_dir / "chunks"

        # 1. Fetch DB Job Record
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(TTSJob).where(TTSJob.id == job_id))
            job = result.scalar_one_or_none()
            if not job:
                logger.error(f"Job {job_id} not found in database.")
                return

            # Update DB to PROCESSING
            job.status = JobStatus.PROCESSING.value
            job.checkpoint_dir = str(checkpoint_dir)
            await db.commit()

            # Retrieve Voice Profile reference audio if specified
            speaker_wav_path: Optional[str] = None
            ref_embedding: Optional[np.ndarray] = None
            speaker_name = "EchoVoice Voice"

            if job.voice_profile_id:
                vp_res = await db.execute(
                    select(VoiceProfile).where(VoiceProfile.id == job.voice_profile_id)
                )
                voice_profile = vp_res.scalar_one_or_none()
                if voice_profile:
                    speaker_name = voice_profile.name
                    speaker_wav_path = voice_profile.reference_audio_path
                    if voice_profile.embedding_path and Path(voice_profile.embedding_path).exists():
                        ref_embedding = np.load(voice_profile.embedding_path)

        try:
            # 2. Check for Existing Checkpoint or Initialize Chunker
            checkpoint_data = self._load_checkpoint_json(checkpoint_file)

            if checkpoint_data:
                logger.info(
                    f"Resuming job {job_id} from checkpoint: "
                    f"{checkpoint_data.get('completed_chunks', 0)}/{checkpoint_data.get('total_chunks', 0)} chunks"
                )
                raw_chunks_meta = checkpoint_data["chunks"]
                total_chunks = checkpoint_data["total_chunks"]
            else:
                # Fresh job: partition script into ~250 character acoustic units
                script_chunks = chunker.chunk_text(job.script_text)
                total_chunks = len(script_chunks)
                raw_chunks_meta = [
                    {
                        "index": c.index,
                        "text": c.text,
                        "pause_after_ms": c.pause_after_ms,
                        "char_count": c.char_count,
                        "audio_file": f"chunk_{c.index:04d}.wav",
                        "status": "PENDING"
                    }
                    for c in script_chunks
                ]

                checkpoint_data = {
                    "job_id": job_id,
                    "status": JobStatus.PROCESSING.value,
                    "total_chunks": total_chunks,
                    "completed_chunks": 0,
                    "chunks": raw_chunks_meta,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                }
                self._write_checkpoint_json(checkpoint_file, checkpoint_data)

                # Update total chunks in DB
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(TTSJob)
                        .where(TTSJob.id == job_id)
                        .values(total_chunks=total_chunks)
                    )
                    await db.commit()

            # 3. Process Chunks (with Checkpoint / Resume)
            completed_count = 0
            for idx, c_meta in enumerate(raw_chunks_meta):
                # Cancellation Check
                if job_id in self._cancelled_jobs:
                    logger.info(f"Job {job_id} cancelled by user during chunk {idx}.")
                    checkpoint_data["status"] = JobStatus.CANCELLED.value
                    checkpoint_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                    self._write_checkpoint_json(checkpoint_file, checkpoint_data)
                    async with AsyncSessionLocal() as db:
                        await db.execute(
                            update(TTSJob)
                            .where(TTSJob.id == job_id)
                            .values(status=JobStatus.CANCELLED.value)
                        )
                        await db.commit()
                    return

                chunk_audio_path = chunks_dir / c_meta["audio_file"]

                # If chunk was already synthesized and verified, skip! (RESUME FEATURE)
                if chunk_audio_path.exists() and chunk_audio_path.stat().st_size > 1000:
                    completed_count += 1
                    c_meta["status"] = "COMPLETED"
                    continue

                # Synthesize chunk speech
                chunk_text = c_meta["text"]
                tts_output, _ = await self.tts_service.generate_speech(
                    text=chunk_text,
                    speaker_wav=speaker_wav_path,
                    language=job.target_language,
                    engine_name=job.engine,
                    output_filename=None
                )

                # Save synthesized audio directly into chunk repository
                sf.write(
                    str(chunk_audio_path),
                    tts_output.audio,
                    tts_output.sample_rate,
                    format="WAV"
                )

                completed_count += 1
                c_meta["status"] = "COMPLETED"
                progress_pct = round((completed_count / max(1, total_chunks)) * 100.0, 1)

                # Update Checkpoint State Atomically
                checkpoint_data["completed_chunks"] = completed_count
                checkpoint_data["progress"] = progress_pct
                checkpoint_data["updated_at"] = datetime.now(timezone.utc).isoformat()
                self._write_checkpoint_json(checkpoint_file, checkpoint_data)

                # Update Database Progress
                async with AsyncSessionLocal() as db:
                    await db.execute(
                        update(TTSJob)
                        .where(TTSJob.id == job_id)
                        .values(
                            completed_chunks=completed_count,
                            progress=progress_pct
                        )
                    )
                    await db.commit()

            # 4. Assembly & Stitching
            logger.info(f"All {total_chunks} chunks synthesized. Assembling and stitching...")
            chunk_audio_list: List[np.ndarray] = []
            pauses_list: List[int] = []
            tts_sr = 24_000

            for c_meta in raw_chunks_meta:
                c_path = chunks_dir / c_meta["audio_file"]
                data, sr = sf.read(str(c_path), dtype="float32")
                chunk_audio_list.append(data)
                pauses_list.append(c_meta.get("pause_after_ms", 400))
                tts_sr = sr

            stitched_audio = audio_stitcher.stitch(
                chunks=chunk_audio_list,
                pauses_ms=pauses_list,
                sr=tts_sr
            )

            # 5. YouTube Audio Mastering Chain
            export_dir = settings.EXPORTS_DIR / job_id
            export_dir.mkdir(parents=True, exist_ok=True)

            mastering_result = youtube_mastering_engine.master(
                audio=stitched_audio,
                source_sr=tts_sr,
                output_dir=export_dir,
                base_filename=f"job_{job_id}",
                title=f"EchoVoice Long-Form #{job_id[:8]}",
                speaker_name=speaker_name
            )

            # 6. Quality Control (QC) Audit
            qc_report = audio_qc_evaluator.evaluate(
                audio=mastering_result.audio,
                sr=mastering_result.sample_rate,
                target_script=job.script_text,
                ref_embedding=ref_embedding
            )

            # 7. Finalize Database Record
            wav_path = str(mastering_result.export_paths.get("wav_24bit_48k", ""))
            mp3_path = str(mastering_result.export_paths.get("mp3_320k", ""))
            flac_path = str(mastering_result.export_paths.get("flac", ""))

            audio_report_payload = {
                "integrated_lufs": mastering_result.integrated_lufs,
                "true_peak_db": mastering_result.true_peak_db,
                "duration_seconds": mastering_result.duration_sec,
                "snr_db": qc_report.snr_db,
                "clipping_ratio": qc_report.clipping_ratio,
                "qc_passed": qc_report.passed,
                "qc_warnings": qc_report.warnings,
                "steps_applied": mastering_result.steps_applied
            }

            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(TTSJob)
                    .where(TTSJob.id == job_id)
                    .values(
                        status=JobStatus.COMPLETED.value,
                        progress=100.0,
                        completed_chunks=total_chunks,
                        output_wav_path=wav_path,
                        output_mp3_path=mp3_path,
                        output_m4a_path=flac_path,
                        wer_score=qc_report.wer,
                        similarity_score=qc_report.speaker_similarity,
                        audio_report=audio_report_payload
                    )
                )
                await db.commit()

            # Finalize Checkpoint
            checkpoint_data["status"] = JobStatus.COMPLETED.value
            checkpoint_data["progress"] = 100.0
            checkpoint_data["completed_chunks"] = total_chunks
            checkpoint_data["updated_at"] = datetime.now(timezone.utc).isoformat()
            self._write_checkpoint_json(checkpoint_file, checkpoint_data)

            logger.info(
                f"Long-form job {job_id} successfully completed! "
                f"Duration: {mastering_result.duration_sec:.1f}s | "
                f"Loudness: {mastering_result.integrated_lufs:.1f} LUFS | QC: {qc_report.passed}"
            )

        except asyncio.CancelledError:
            logger.info(f"Job {job_id} execution cancelled.")
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(TTSJob)
                    .where(TTSJob.id == job_id)
                    .values(status=JobStatus.CANCELLED.value)
                )
                await db.commit()

        except Exception as exc:
            logger.error(f"Error executing long-form job {job_id}: {exc}", exc_info=True)
            async with AsyncSessionLocal() as db:
                await db.execute(
                    update(TTSJob)
                    .where(TTSJob.id == job_id)
                    .values(
                        status=JobStatus.FAILED.value,
                        error_message=str(exc)
                    )
                )
                await db.commit()
            if checkpoint_data:
                checkpoint_data["status"] = JobStatus.FAILED.value
                checkpoint_data["error"] = str(exc)
                self._write_checkpoint_json(checkpoint_file, checkpoint_data)
        finally:
            self._active_tasks.pop(job_id, None)
            self._cancelled_jobs.discard(job_id)


job_runner = LongFormJobRunner()
