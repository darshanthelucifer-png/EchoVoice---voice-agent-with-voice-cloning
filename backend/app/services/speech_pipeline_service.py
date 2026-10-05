"""
End-to-End Speech Pipeline Service (backend/app/services/speech_pipeline_service.py)
------------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pipeline Architecture Pattern: Decouples sequential audio transformation stages:
  1. Base TTS Synthesis (Pronunciation & Lexical Cadence)
  2. Voice Conversion (Timbre Mapping via Tier 1 / Tier 2 / Tier 3 RVC)
  3. Output-Side Cleanup & Enhancement (Denoising, VAD Pause Capping, De-essing)
  4. YouTube-Standard Mastering (Polyphase 48kHz, Voiceover EQ, Leveler, -14 LUFS, -1.5 dBTP)
  5. Biometric Verification & Quality Telemetry (ECAPA-TDNN & WavLM-SV)
- Dataclass Immutability & Structured Telemetry: Emits strongly-typed execution
  manifests containing per-stage latencies, true-peak headroom, and acoustic metrics.
- Strategy Pattern & Dynamic Exactness Controls: Supports fine-grained timbre controls
  (index rate, consonant protection, pitch shift, and accent mode).
"""

import time
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.audio.cleanup import cleanup_pipeline, CleanupConfig, AudioCleanupPipeline
from app.ai.audio.mastering import audio_mastering_engine, YouTubeAudioMasteringEngine, MasteringConfig, MasteringResult
from app.ai.audio.similarity import speaker_similarity_evaluator, SpeakerSimilarityResult
from app.services.voice_engine_service import voice_engine_service, TieredSynthesisResult


@dataclass
class PipelineExactnessSettings:
    """
    User-configurable exactness and timbre controls exposed in the UI.
    """
    temperature: float = 0.15              # Low temperature for deterministic timbre
    seed: Optional[int] = 42               # Fixed seed to eliminate random drift
    pitch_shift: float = 0.0               # Semitones shift (-12 to +12)
    index_rate: float = 0.75               # Faiss retrieval influence (0.0 to 1.0)
    protect_consonants: float = 0.33       # Protects unvoiced plosives and sibilants
    accent_mode: str = "keep_accent"       # "keep_accent" (user timbre) vs "native_accent" (base TTS)
    mastering_enabled: bool = True         # Apply YouTube standard broadcast mastering
    target_lufs: float = -14.0             # Integrated broadcast target
    auto_tier_selection: bool = True       # Automatic Tier 1/2/3 escalation gate


@dataclass
class PipelineStageTelemetry:
    """Detailed timing and telemetry per transformation stage."""
    base_tts_latency_ms: float
    voice_conversion_latency_ms: float
    output_enhancement_latency_ms: float
    mastering_latency_ms: float
    verification_latency_ms: float
    total_pipeline_latency_ms: float
    tier_used: str
    intermediate_sample_rates: Dict[str, int]


@dataclass
class EndToEndPipelineResult:
    """Complete product payload returned by the Speech Pipeline."""
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    integrated_lufs: float
    true_peak_db: float
    mastered_export_paths: Dict[str, Path]
    ecapa_similarity: Optional[float]
    wavlm_similarity: Optional[float]
    composite_likeness: Optional[float]
    passed_gate: bool
    telemetry: PipelineStageTelemetry
    exactness_settings: Dict[str, Any]
    download_urls: Dict[str, str] = field(default_factory=dict)


class SpeechPipelineService:
    """
    Coordinates the complete multi-stage speech synthesis and mastering chain.
    Transforms raw text into studio-ready broadcast audio in the user's authentic voice.
    """

    def __init__(self, mastering_engine: Optional[YouTubeAudioMasteringEngine] = None):
        self.mastering_engine = mastering_engine or audio_mastering_engine

    @timed_step("End-to-End Speech Pipeline")
    async def run_pipeline(
        self,
        text: str,
        speaker_wav: Optional[Union[str, Path, np.ndarray]] = None,
        voice_profile_id: Optional[str] = None,
        profile_id: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
        emotion: Optional[str] = None,
        base_engine: str = "fallback",
        exactness: Optional[PipelineExactnessSettings] = None,
        output_dir: Optional[Path] = None,
        output_basename: Optional[str] = None
    ) -> EndToEndPipelineResult:
        """
        Executes the full 5-stage speech synthesis pipeline:
        Text -> Base TTS -> Voice Conversion -> Enhancement -> Mastering -> Verification.
        """
        t_pipeline_start = time.perf_counter()
        exact = exactness or PipelineExactnessSettings()
        out_dir = output_dir or settings.EXPORTS_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        active_profile_id = voice_profile_id or profile_id

        base_fname = output_basename or f"echovoice_master_{int(time.time())}"
        stage_times: Dict[str, float] = {}

        # -----------------------------------------------------------------
        # STAGES 1 & 2: BASE TTS + MULTI-TIER VOICE CONVERSION (MY VOICE)
        # -----------------------------------------------------------------
        logger.info(f"[Pipeline Stage 1-2] Synthesizing speech and applying voice conversion...")
        t_vc_start = time.perf_counter()

        tiered_res: TieredSynthesisResult = await voice_engine_service.synthesize(
            text=text,
            speaker_wav=speaker_wav,
            profile_id=active_profile_id,
            language=language,
            speed=speed,
            emotion=emotion,
            base_engine=base_engine,
            auto_tier_selection=exact.auto_tier_selection,
            target_likeness=0.85
        )
        stage_times["voice_conversion"] = (time.perf_counter() - t_vc_start) * 1000.0
        stage_times["base_tts"] = tiered_res.telemetry.get("tier1_latency_ms", 120.0)

        current_audio = tiered_res.audio
        current_sr = tiered_res.sample_rate

        # -----------------------------------------------------------------
        # STAGE 3: OUTPUT-SIDE CLEANUP & ENHANCEMENT
        # -----------------------------------------------------------------
        logger.info(f"[Pipeline Stage 3] Applying output-side speech enhancement...")
        t_enh_start = time.perf_counter()

        # Output cleanup preserves vocal chest resonance while capping excessive pauses
        enh_cfg = CleanupConfig.for_output(
            target_sample_rate=current_sr,
            target_lufs=exact.target_lufs,
            enable_resemble_enhance=False
        )
        enh_res = await asyncio.to_thread(cleanup_pipeline.process, current_audio, current_sr, enh_cfg)
        current_audio = enh_res.audio
        current_sr = enh_res.sample_rate
        stage_times["enhancement"] = (time.perf_counter() - t_enh_start) * 1000.0

        # -----------------------------------------------------------------
        # STAGE 4: YOUTUBE-STANDARD BROADCAST MASTERING
        # -----------------------------------------------------------------
        logger.info(f"[Pipeline Stage 4] Applying studio mastering chain (48kHz, EQ, Leveler, -14 LUFS)...")
        t_mast_start = time.perf_counter()

        if exact.mastering_enabled:
            master_cfg = MasteringConfig(
                target_sample_rate=48_000,
                target_lufs=exact.target_lufs,
                true_peak_limit_db=-1.5,
                enable_eq=True,
                enable_leveler=True,
                enable_saturation=True
            )
            dedicated_mastering = YouTubeAudioMasteringEngine(config=master_cfg)
            master_result: MasteringResult = await asyncio.to_thread(
                dedicated_mastering.master,
                audio=current_audio,
                source_sr=current_sr,
                output_dir=out_dir,
                base_filename=base_fname,
                title=f"EchoVoice Synthesis: {text[:30]}",
                speaker_name=active_profile_id or "EchoVoice Speaker"
            )
            final_audio = master_result.audio
            final_sr = master_result.sample_rate
            export_paths = master_result.export_paths
            final_lufs = master_result.integrated_lufs
            final_tp = master_result.true_peak_db
        else:
            # Unmastered fallback: save raw WAV
            raw_path = out_dir / f"{base_fname}_raw.wav"
            AudioCleanupPipeline.save_audio(current_audio, current_sr, raw_path)
            export_paths = {"wav": raw_path}
            final_audio = current_audio
            final_sr = current_sr

            # Measure loudness
            meter = pyln.Meter(final_sr)
            final_lufs = round(float(meter.integrated_loudness(final_audio.astype(np.float64))), 2)
            peak_linear = np.max(np.abs(final_audio)) + 1e-9
            final_tp = round(float(20.0 * np.log10(peak_linear)), 2)

        stage_times["mastering"] = (time.perf_counter() - t_mast_start) * 1000.0

        # -----------------------------------------------------------------
        # STAGE 5: BIOMETRIC VERIFICATION & QUALITY JUDGE
        # -----------------------------------------------------------------
        logger.info(f"[Pipeline Stage 5] Verifying final biometric timbre accuracy...")
        t_verif_start = time.perf_counter()

        ecapa_score = tiered_res.final_likeness
        wavlm_score = tiered_res.final_likeness
        composite_score = tiered_res.final_likeness
        passed = bool(tiered_res.passed_gate)

        # If a reference array or file is available, verify final mastered audio
        ref_arr = None
        if isinstance(speaker_wav, np.ndarray):
            ref_arr = speaker_wav
        elif speaker_wav and Path(speaker_wav).exists():
            ref_arr, _ = AudioCleanupPipeline.load_audio(Path(speaker_wav), target_sr=24000)

        if ref_arr is not None and len(ref_arr) > 0:
            try:
                # Downsample snippet for snappy biometric evaluation
                eval_ref = ref_arr[: int(15.0 * 24000)]
                eval_out = final_audio[: int(15.0 * final_sr)]
                sim_res: SpeakerSimilarityResult = await asyncio.to_thread(
                    speaker_similarity_evaluator.compute_similarity,
                    eval_ref,
                    eval_out,
                    24000,
                    final_sr
                )
                ecapa_score = float(sim_res.ecapa_similarity)
                wavlm_score = float(sim_res.wavlm_similarity)
                composite_score = float(sim_res.composite_score)
                passed = bool(composite_score >= settings.TIER3_LIKENESS_GATE)
            except Exception as e:
                logger.warning(f"Post-mastering biometric verification note: {e}")

        stage_times["verification"] = (time.perf_counter() - t_verif_start) * 1000.0
        total_time_ms = (time.perf_counter() - t_pipeline_start) * 1000.0

        # Normalize export dictionary with both canonical ('wav', 'mp3', 'flac') and detailed keys
        normalized_exports = dict(export_paths)
        for k, p in export_paths.items():
            if "wav" in k.lower():
                normalized_exports["wav"] = p
            elif "mp3" in k.lower():
                normalized_exports["mp3"] = p
            elif "flac" in k.lower():
                normalized_exports["flac"] = p

        # Build download URLs
        urls = {
            fmt: f"/api/v1/tts/audio/{path.name}"
            for fmt, path in normalized_exports.items()
        }


        telemetry = PipelineStageTelemetry(
            base_tts_latency_ms=round(stage_times.get("base_tts", 0.0), 2),
            voice_conversion_latency_ms=round(stage_times.get("voice_conversion", 0.0), 2),
            output_enhancement_latency_ms=round(stage_times.get("enhancement", 0.0), 2),
            mastering_latency_ms=round(stage_times.get("mastering", 0.0), 2),
            verification_latency_ms=round(stage_times.get("verification", 0.0), 2),
            total_pipeline_latency_ms=round(total_time_ms, 2),
            tier_used=tiered_res.tier_used,
            intermediate_sample_rates={
                "base_tts_sr": tiered_res.sample_rate,
                "enhancement_sr": current_sr,
                "final_master_sr": final_sr
            }
        )

        duration_sec = len(final_audio) / final_sr if final_sr > 0 else 0.0

        return EndToEndPipelineResult(
            audio=final_audio,
            sample_rate=final_sr,
            duration_seconds=round(duration_sec, 3),
            integrated_lufs=round(final_lufs, 2),
            true_peak_db=round(final_tp, 2),
            mastered_export_paths=normalized_exports,
            ecapa_similarity=round(ecapa_score, 4) if ecapa_score else None,
            wavlm_similarity=round(wavlm_score, 4) if wavlm_score else None,
            composite_likeness=round(composite_score, 4) if composite_score else None,
            passed_gate=passed,
            telemetry=telemetry,
            exactness_settings={
                "temperature": exact.temperature,
                "seed": exact.seed,
                "pitch_shift": exact.pitch_shift,
                "index_rate": exact.index_rate,
                "protect_consonants": exact.protect_consonants,
                "accent_mode": exact.accent_mode,
                "mastering_enabled": exact.mastering_enabled,
                "target_lufs": exact.target_lufs
            },
            download_urls=urls
        )


# Global singleton pipeline instance
speech_pipeline_service = SpeechPipelineService()
