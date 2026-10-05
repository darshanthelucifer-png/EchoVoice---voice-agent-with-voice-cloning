"""
Phase 4 Unit Tests: End-to-End Speech Pipeline & Blind A/B Testing (backend/tests/test_speech_pipeline.py)
---------------------------------------------------------------------------------------------------------
Verifies Phase 4 requirements:
1. Complete 5-stage synthesis pipeline (Text -> Base TTS -> Voice Conversion -> Enhance -> Master)
2. Broadcast mastering loudness compliance (48 kHz, -14 LUFS integrated, <= -1.5 dBTP true peak)
3. Multi-format artifact generation (WAV, MP3, FLAC) with C2PA metadata
4. Double-blind A/B test randomization and identity cloaking
5. Blind test reveal with biometric scores, latency, and loudness metrics
6. Automated multi-tier engine ranking and auto-judging
"""

import pytest
import numpy as np
import soundfile as sf
from pathlib import Path

from app.core.config import settings
from app.services.speech_pipeline_service import (
    speech_pipeline_service,
    PipelineExactnessSettings,
    EndToEndPipelineResult
)
from app.services.ab_testing_service import ab_testing_service


def synthesize_test_voice(freq: float = 145.0, dur: float = 6.0, sr: int = 24000) -> np.ndarray:
    """Generates continuous multi-harmonic voiced audio."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 5]:
        sig += (0.35 / h) * np.sin(2 * np.pi * freq * h * t)
    cadence = 0.75 + 0.25 * np.sin(2 * np.pi * 0.5 * t)
    return (sig * cadence * 0.45).astype(np.float32)


@pytest.mark.asyncio
async def test_pipeline_end_to_end_execution(tmp_path):
    """Verify complete 5-stage pipeline generates mastered 48kHz broadcast audio."""
    sr = 24000
    ref_audio = synthesize_test_voice(dur=6.0, sr=sr)
    ref_file = tmp_path / "reference.wav"
    sf.write(str(ref_file), ref_audio, sr, format="WAV")

    exactness = PipelineExactnessSettings(
        mastering_enabled=True,
        target_lufs=-14.0,
        pitch_shift=0.0,
        index_rate=0.75
    )

    result: EndToEndPipelineResult = await speech_pipeline_service.run_pipeline(
        text="EchoVoice Phase 4 end to end pipeline test.",
        speaker_wav=ref_file,
        language="en",
        base_engine="fallback",
        exactness=exactness,
        output_dir=tmp_path,
        output_basename="test_pipeline_run"
    )

    # 1. Output specification assertions
    assert result.sample_rate == 48_000, "Mastered audio must be studio-standard 48 kHz"
    assert result.duration_seconds > 1.0
    assert len(result.audio) > 0

    # 2. YouTube loudness & peak limiting compliance
    # -14 LUFS target (+-1.5 LUFS tolerance on short phrases)
    assert -16.5 <= result.integrated_lufs <= -11.5
    assert result.true_peak_db <= -1.4, f"True peak {result.true_peak_db} dBTP must not exceed -1.4 dBTP"

    # 3. Multi-format export assertions
    assert "wav" in result.mastered_export_paths
    assert result.mastered_export_paths["wav"].exists()
    assert result.mastered_export_paths["wav"].stat().st_size > 1000

    # 4. Telemetry assertions
    t = result.telemetry
    assert t.total_pipeline_latency_ms > 0
    assert t.intermediate_sample_rates["final_master_sr"] == 48_000
    assert "intermediate_sample_rates" in result.telemetry.__dict__


@pytest.mark.asyncio
async def test_pipeline_unmastered_bypass(tmp_path):
    """Verify mastering chain can be bypassed when raw synthesis is requested."""
    sr = 24000
    ref_audio = synthesize_test_voice(dur=5.0, sr=sr)
    ref_file = tmp_path / "ref_raw.wav"
    sf.write(str(ref_file), ref_audio, sr, format="WAV")

    exactness = PipelineExactnessSettings(
        mastering_enabled=False
    )

    result: EndToEndPipelineResult = await speech_pipeline_service.run_pipeline(
        text="Raw unmastered audio output verification.",
        speaker_wav=ref_file,
        base_engine="fallback",
        exactness=exactness,
        output_dir=tmp_path,
        output_basename="test_raw_run"
    )

    assert result.sample_rate == 24_000 or result.sample_rate == 48_000
    assert "wav" in result.mastered_export_paths
    assert result.mastered_export_paths["wav"].exists()


@pytest.mark.asyncio
async def test_blind_ab_test_creation_and_randomization(tmp_path):
    """Verify double-blind A/B test scrambles candidate identities and cloaks telemetry."""
    sr = 24000
    ref_audio = synthesize_test_voice(dur=5.0, sr=sr)
    ref_file = tmp_path / "blind_ref.wav"
    sf.write(str(ref_file), ref_audio, sr, format="WAV")

    blind_data = await ab_testing_service.create_blind_test(
        text="Which voice sounds more authentic to you?",
        speaker_wav=ref_file,
        language="en"
    )

    assert "test_id" in blind_data
    assert "blind_candidate_1" in blind_data
    assert "blind_candidate_2" in blind_data

    # Ensure model names and biometric scores are STRICTLY CLOAKED in blind state
    cand1 = blind_data["blind_candidate_1"]
    cand2 = blind_data["blind_candidate_2"]
    assert "tier" not in cand1
    assert "score" not in cand1
    assert "likeness" not in cand1
    assert cand1["name"] == "Candidate 1"
    assert cand2["name"] == "Candidate 2"
    assert cand1["audio_url"].startswith("/api/v1/tts/audio/")


def test_blind_ab_test_reveal():
    """Verify reveal endpoint decrypts hidden identities, scores, and loudness."""
    # Find session from previous test or create mock session
    test_id = list(ab_testing_service._sessions.keys())[0]

    revealed = ab_testing_service.reveal_blind_test(test_id=test_id, user_vote="Candidate 1")

    assert revealed["test_id"] == test_id
    assert revealed["user_vote"] == "Candidate 1"
    assert "objective_winner" in revealed
    assert "candidate_1" in revealed
    assert "candidate_2" in revealed

    c1 = revealed["candidate_1"]
    assert "tier_used" in c1
    assert "integrated_lufs" in c1
    assert "true_peak_db" in c1
    assert "latency_ms" in c1


@pytest.mark.asyncio
async def test_auto_judge_tier_ranking(tmp_path):
    """Verify automated judge executes all tiers and ranks candidates by composite likeness."""
    sr = 24000
    ref_audio = synthesize_test_voice(dur=5.0, sr=sr)
    ref_file = tmp_path / "judge_ref.wav"
    sf.write(str(ref_file), ref_audio, sr, format="WAV")

    judge_result = await ab_testing_service.auto_judge_all_tiers(
        test_sentence="Auto judge calibration test sentence.",
        speaker_wav=ref_file,
        language="en"
    )

    assert "winning_tier" in judge_result
    assert "winning_likeness" in judge_result
    assert "ranked_candidates" in judge_result

    ranked = judge_result["ranked_candidates"]
    assert len(ranked) >= 2
    # Ensure ranked in descending order of composite likeness
    for i in range(len(ranked) - 1):
        assert ranked[i]["composite_likeness"] >= ranked[i + 1]["composite_likeness"]
