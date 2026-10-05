"""
Voice Profile Service (backend/app/services/voice_profile_service.py)
---------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Ethical Consent Gate: Enforces strict legal and ethical consent auditing
  with UTC timestamps before voice biometric processing.
- Biometric Feature Extraction: Computes speaker embeddings and identifies the
  top 3 cleanest acoustic reference clips for stable zero-shot voice cloning.
- Secure Resource Cleanup: Completely wipes audio files, extracted clips, and
  embedding tensors from the filesystem upon profile deletion (privacy compliance).
- Async/Await with ORM Cascades: Non-blocking database transactions.
"""

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from fastapi import HTTPException, status

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.models.voice_profile import VoiceProfile
from app.models.user import User
from app.schemas.voice_profile import VoiceProfileCreate
from app.ai.audio.cleanup import (
    AudioCleanupPipeline,
    CleanupConfig,
    CleanupResult,
    cleanup_pipeline,
)
from app.ai.audio.quality import quality_analyzer, AudioQualityReport
from app.services.base import BaseService


class VoiceProfileService(BaseService[VoiceProfile]):
    """
    Manages voice enrollment, consent auditing, speaker embedding extraction,
    and profile lifecycle.
    """
    def __init__(self):
        super().__init__(VoiceProfile)

    def _analyze_candidate_clip(self, clip: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Analyzes acoustic suitability of a candidate reference clip:
        - SNR (dB)
        - Clipping ratio
        - Voiced pitch presence (autocorrelation in 75-350 Hz)
        - Mean F0 pitch
        - Spectral dynamics (phonetic richness)
        """
        rms = float(np.sqrt(np.mean(clip ** 2) + 1e-12))
        peak = float(np.max(np.abs(clip)))
        clipping_ratio = float(np.sum(np.abs(clip) >= 0.99)) / max(1, len(clip))

        # Local SNR proxy (50ms frames)
        frame_len = int(sr * 0.05)
        hop = int(sr * 0.025)
        num_frames = (len(clip) - frame_len) // hop + 1
        if num_frames > 4:
            f_rms = [np.sqrt(np.mean(clip[i * hop : i * hop + frame_len] ** 2) + 1e-12) for i in range(num_frames)]
            noise_fl = float(np.percentile(f_rms, 15))
            speech_fl = float(np.percentile(f_rms, 85))
            snr = max(0.0, 20.0 * np.log10(max(speech_fl, 1e-6) / max(noise_fl, 1e-6)))
        else:
            snr = 15.0

        # Pitch estimation using autocorrelation
        chunk = clip[:min(len(clip), sr * 2)]
        corr = np.correlate(chunk, chunk, mode="full")
        corr = corr[len(corr) // 2 :]
        min_lag = int(sr / 350)
        max_lag = int(sr / 75)
        if max_lag < len(corr):
            lag = np.argmax(corr[min_lag:max_lag]) + min_lag
        else:
            lag = min_lag
        f0 = sr / lag if lag > 0 else 130.0
        peak_ratio = float(corr[lag] / max(corr[0], 1e-9)) if lag < len(corr) else 0.0

        # Quality score penalizes clipping and unvoiced noise
        quality_score = snr * 0.5 + peak_ratio * 30.0 - (clipping_ratio * 1000.0)

        return {
            "rms": rms,
            "peak": peak,
            "snr": snr,
            "clipping_ratio": clipping_ratio,
            "f0": f0,
            "voiced_ratio": peak_ratio,
            "quality_score": quality_score
        }

    def _extract_speaker_embedding(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Extracts a normalized neural speaker embedding using SpeechBrain ECAPA-TDNN.
        Falls back to spectral moment heuristic if neural model is offline.
        """
        try:
            from app.ai.audio.similarity import speaker_similarity_evaluator
            return speaker_similarity_evaluator.extract_ecapa_embedding(audio, sr)
        except Exception as e:
            logger.warning(f"Neural speaker embedding fallback to spectral moment: {e}")
            from scipy import signal
            f, t, sxx = signal.spectrogram(audio, sr, nperseg=512, noverlap=256)
            log_sxx = np.log(sxx + 1e-6)
            mean_spec = np.mean(log_sxx, axis=1)
            std_spec = np.std(log_sxx, axis=1)
            raw_feat = np.concatenate([mean_spec, std_spec])
            interp_feat = np.interp(
                np.linspace(0, len(raw_feat) - 1, 192),
                np.arange(len(raw_feat)),
                raw_feat
            ).astype(np.float32)
            norm = np.linalg.norm(interp_feat) + 1e-12
            return (interp_feat / norm).astype(np.float32)

    def _extract_best_reference_clips(
        self,
        audio: np.ndarray,
        sr: int,
        target_clips: int = 3,
        clip_duration_sec: float = 6.0
    ) -> List[np.ndarray]:
        """
        Automatically selects the best 3 to 5 reference clips with high SNR,
        minimal clipping, and maximum pitch/energy diversity.
        """
        clip_samples = int(clip_duration_sec * sr)
        if len(audio) <= clip_samples:
            return [audio]

        step = int(clip_samples * 0.6)  # 40% overlap maximum
        candidates = []

        for start in range(0, len(audio) - clip_samples + 1, step):
            sub = audio[start : start + clip_samples]
            metrics = self._analyze_candidate_clip(sub, sr)
            if metrics["clipping_ratio"] < 0.005 and metrics["rms"] > 0.015:
                candidates.append((metrics["quality_score"], metrics["f0"], sub))

        if not candidates:
            # Fallback to linear slices
            return [audio[i : i + clip_samples] for i in range(0, min(len(audio), clip_samples * target_clips), clip_samples)]

        # Sort descending by quality
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Greedy maximal diversity selection (anchor highest-quality, then diversify by pitch)
        selected = [candidates[0]]
        for cand in candidates[1:]:
            if len(selected) >= target_clips:
                break
            min_f0_diff = min(abs(cand[1] - s[1]) for s in selected)
            if min_f0_diff > 10.0 or len(candidates) < target_clips * 2:
                selected.append(cand)

        # Fill remaining slots if needed
        if len(selected) < target_clips:
            for cand in candidates:
                if cand not in selected:
                    selected.append(cand)
                    if len(selected) >= target_clips:
                        break

        return [s[2] for s in selected]

    @timed_step("Voice Profile Enrollment Pipeline")
    async def enroll_profile(
        self,
        db: AsyncSession,
        user: User,
        audio_bytes: bytes,
        name: str,
        description: Optional[str] = None,
        consent_given: bool = False,
        cleanup_cfg: Optional[CleanupConfig] = None
    ) -> Tuple[VoiceProfile, CleanupResult]:
        """
        Full enrollment workflow:
        1. Consent gate validation
        2. Light reference cleanup (no room tone, 50 Hz HPF, gentle denoise)
        3. Neural ECAPA-TDNN speaker embedding extraction
        4. Best 3-5 reference clips selection (diverse pitch + high SNR)
        5. Tier eligibility determination (Tier 1-2 vs Tier 3 unlocked)
        6. Database record persistence
        """
        # --- ETHICAL CONSENT GATE ---
        if not consent_given:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Consent Required: You must explicitly confirm that this is your own voice "
                    "or that you have written permission from the speaker before creating a voice profile."
                )
            )

        consent_timestamp = datetime.now(timezone.utc)

        # 1. Decode audio
        raw_audio, sr = AudioCleanupPipeline.load_audio(audio_bytes)
        if len(raw_audio) < sr * 5:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Recording is too short. Please provide at least 5 to 20 seconds of speech."
            )

        # 2. Run Light Cleanup Pipeline for Reference Audio
        cfg = cleanup_cfg or CleanupConfig.for_reference()
        result = cleanup_pipeline.process(raw_audio, sr, cfg)

        # 3. Create Profile Directory
        import uuid
        profile_id = str(uuid.uuid4())
        profile_dir = settings.VOICE_PROFILES_DIR / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = profile_dir / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

        # Save raw and clean reference audio
        raw_path = profile_dir / "raw_recording.wav"
        ref_path = profile_dir / "reference.wav"
        AudioCleanupPipeline.save_audio(raw_audio, sr, raw_path)
        AudioCleanupPipeline.save_audio(result.audio, result.sample_rate, ref_path)

        # 4. Extract Neural Speaker Embedding & Best 3-5 Reference Clips
        embedding = self._extract_speaker_embedding(result.audio, result.sample_rate)
        emb_path = profile_dir / "speaker_embedding.npy"
        np.save(str(emb_path), embedding)

        # Target 3 to 5 clips depending on speech duration
        speech_sec = result.cleaned_report.speech_duration_seconds
        target_clips = 5 if speech_sec >= 30.0 else (4 if speech_sec >= 20.0 else 3)
        clips = self._extract_best_reference_clips(result.audio, result.sample_rate, target_clips=target_clips)
        for idx, clip in enumerate(clips, 1):
            AudioCleanupPipeline.save_audio(clip, result.sample_rate, clips_dir / f"clip_{idx}.wav")

        # 5. Tier Eligibility Calculation
        total_duration_sec = result.cleaned_report.duration_seconds
        tier_status = "tier_3_eligible" if total_duration_sec >= 180.0 else "tier_1_2"
        tier_notice = (
            "Tier 3 eligible! Audio length >= 3 minutes. Ready for dedicated RVC voice model training."
            if total_duration_sec >= 180.0
            else f"Tier 1–2 unlocked ({total_duration_sec:.1f}s audio). Record 3+ minutes to unlock Tier 3 exact trained voice model."
        )

        quality_data = result.cleaned_report.to_dict()
        quality_data.update({
            "tier_status": tier_status,
            "tier_notice": tier_notice,
            "selected_clips_count": len(clips),
            "clean_snr_db": result.cleaned_report.snr_db,
            "raw_snr_db": result.raw_report.snr_db,
            "duration_minutes": round(total_duration_sec / 60.0, 2)
        })

        # 6. Check if user already has a default profile
        existing_profiles = await self.get_user_profiles(db, user.id)
        is_first_profile = len(existing_profiles) == 0

        # 7. Save DB Record
        profile = VoiceProfile(
            id=profile_id,
            user_id=user.id,
            name=name.strip(),
            description=description.strip() if description else None,
            reference_audio_path=str(ref_path),
            raw_audio_path=str(raw_path),
            speaker_embedding_path=str(emb_path),
            consent_given=True,
            consent_timestamp=consent_timestamp,
            quality_metrics=quality_data,
            is_default=is_first_profile
        )

        db.add(profile)
        await db.flush()
        await db.refresh(profile)

        result.output_path = ref_path
        return profile, result

    async def get_user_profiles(self, db: AsyncSession, user_id: str) -> List[VoiceProfile]:
        """Fetch all profiles owned by user."""
        res = await db.execute(
            select(VoiceProfile)
            .where(VoiceProfile.user_id == user_id)
            .order_by(VoiceProfile.created_at.desc())
        )
        return list(res.scalars().all())

    async def get_profile_by_id(self, db: AsyncSession, profile_id: str, user_id: Optional[str] = None) -> Optional[VoiceProfile]:
        """Fetch profile verifying ownership if user_id is provided, or by profile ID directly."""
        stmt = select(VoiceProfile).where(VoiceProfile.id == profile_id)
        if user_id:
            stmt = stmt.where(VoiceProfile.user_id == user_id)
        res = await db.execute(stmt)
        return res.scalars().first()

    async def set_default_profile(self, db: AsyncSession, profile_id: str, user_id: str) -> VoiceProfile:
        """Sets selected profile as user's default, resetting others."""
        profile = await self.get_profile_by_id(db, profile_id, user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found."
            )

        # Reset all other user profiles
        await db.execute(
            update(VoiceProfile)
            .where(VoiceProfile.user_id == user_id)
            .values(is_default=False)
        )
        profile.is_default = True
        await db.flush()
        await db.refresh(profile)
        return profile

    async def delete_profile(self, db: AsyncSession, profile_id: str, user_id: str) -> bool:
        """
        Deletes database record and wipes all audio files and embeddings from disk.
        """
        profile = await self.get_profile_by_id(db, profile_id, user_id)
        if not profile:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found."
            )

        # 1. Wipe files from disk
        profile_dir = settings.VOICE_PROFILES_DIR / profile.id
        if profile_dir.exists():
            shutil.rmtree(profile_dir, ignore_errors=True)

        # 2. Delete database record
        await db.delete(profile)
        await db.flush()
        return True


voice_profile_service = VoiceProfileService()
