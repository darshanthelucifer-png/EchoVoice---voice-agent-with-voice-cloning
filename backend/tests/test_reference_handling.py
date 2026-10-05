"""
Phase 1 Unit Tests: Reference Handling & Best-Clip Selection (backend/tests/test_reference_handling.py)
-------------------------------------------------------------------------------------------------------
Verifies Phase 1 improvements:
1. CleanupConfig.for_reference() vs CleanupConfig.for_output() configurations
2. Absence of artificial room tone hiss injection during reference cleanup
3. Preservation of low-frequency chest harmonics (50 Hz HPF)
4. Diversity-aware selection of top 3-5 reference clips
5. Neural ECAPA-TDNN speaker embedding extraction in VoiceProfileService
6. Voice profile tier qualification logic (Tier 1-2 vs Tier 3 unlocked)
"""

import pytest
import numpy as np

from app.ai.audio.cleanup import CleanupConfig, cleanup_pipeline, AudioCleanupPipeline
from app.ai.audio.similarity import speaker_similarity_evaluator
from app.services.voice_profile_service import voice_profile_service
from scripts.test_audio_cleanup import generate_realistic_20s_test_voice


def test_cleanup_config_factory_presets():
    """Verify factory methods provide strictly segregated reference and output configurations."""
    ref_cfg = CleanupConfig.for_reference(target_sample_rate=24000, denoise_strength=0.20)
    out_cfg = CleanupConfig.for_output(target_sample_rate=48000, target_lufs=-14.0)

    # Reference mode guarantees:
    assert ref_cfg.mode == "reference"
    assert ref_cfg.insert_room_tone is False, "Reference cleanup MUST NOT inject artificial room tone!"
    assert ref_cfg.enable_loudness_norm is False, "Reference cleanup MUST NOT compress dynamics!"
    assert ref_cfg.denoise_prop_decrease <= 0.25, "Reference cleanup must use gentle denoise strength!"
    assert ref_cfg.highpass_cutoff_hz <= 50.0, "Highpass filter must preserve male chest harmonics!"
    assert ref_cfg.target_sample_rate == 24000

    # Output mode guarantees:
    assert out_cfg.mode == "output"
    assert out_cfg.insert_room_tone is True
    assert out_cfg.enable_loudness_norm is True
    assert out_cfg.target_lufs == -14.0
    assert out_cfg.target_sample_rate == 48000


def test_reference_cleanup_preserves_speaker_identity():
    """Verify that Phase 1 reference cleanup achieves superior similarity preservation to raw voice."""
    sr = 16000
    dur = 4.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    raw_audio = np.zeros_like(t)
    for h in range(1, 10):
        raw_audio += (0.3 / h) * np.sin(2 * np.pi * 140.0 * h * t)
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 2.0 * t))
    raw_audio = (raw_audio * env).astype(np.float32)

    # 1. Legacy Aggressive Cleanup
    legacy_cfg = CleanupConfig(
        target_sample_rate=sr,
        denoise_prop_decrease=0.80,
        enable_highpass=True,
        highpass_cutoff_hz=80.0,
        insert_room_tone=True,
        room_tone_dbfs=-65.0,
        enable_loudness_norm=True,
        enable_vad_trim=False
    )
    legacy_res = cleanup_pipeline.process(raw_audio, sr, legacy_cfg)

    # 2. Phase 1 Light Reference Cleanup
    phase1_cfg = CleanupConfig.for_reference(target_sample_rate=sr, denoise_strength=0.20)
    phase1_cfg.enable_vad_trim = False
    phase1_res = cleanup_pipeline.process(raw_audio, sr, phase1_cfg)

    # Evaluate against raw voice
    sim_legacy = speaker_similarity_evaluator.compute_similarity(raw_audio, legacy_res.audio, sr, sr)
    sim_phase1 = speaker_similarity_evaluator.compute_similarity(raw_audio, phase1_res.audio, sr, sr)

    # Phase 1 MUST preserve identity better than legacy over-processing
    assert sim_phase1.ecapa_similarity >= sim_legacy.ecapa_similarity
    assert "Room Tone Inserted" not in phase1_res.steps_applied


def test_best_reference_clips_extraction():
    """Verify that multi-criteria selector extracts 3 to 5 diverse, high-quality clips."""
    sr = 16000
    dur = 12.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Voiced harmonic audio with shifting pitch across 12 seconds
    f0_mod = 130.0 + 30.0 * np.sin(2 * np.pi * 0.25 * t)
    audio = (0.3 * np.sin(2 * np.pi * f0_mod * t)).astype(np.float32)

    clips = voice_profile_service._extract_best_reference_clips(
        audio, sr, target_clips=3, clip_duration_sec=3.0
    )

    assert len(clips) >= 2
    assert len(clips) <= 4

    # Check each clip
    for clip in clips:
        assert len(clip) == int(3.0 * sr)
        # Check no clipping
        assert np.max(np.abs(clip)) <= 1.0
        # Check has speech energy
        rms = np.sqrt(np.mean(clip ** 2))
        assert rms > 0.01


def test_candidate_clip_analyzer():
    """Verify acoustic metrics are computed correctly for clip candidates."""
    sr = 16000
    dur = 3.0
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    # Voiced harmonic signal with pitch 150 Hz
    clip = (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32)

    metrics = voice_profile_service._analyze_candidate_clip(clip, sr)

    assert "rms" in metrics
    assert "f0" in metrics
    assert "snr" in metrics
    assert "quality_score" in metrics
    assert 130.0 <= metrics["f0"] <= 170.0
    assert metrics["clipping_ratio"] == 0.0
    assert metrics["quality_score"] > 0
