"""
Evaluation Harness Unit Tests (backend/tests/test_evaluation_harness.py)
-----------------------------------------------------------------------
Verifies Phase 0 Evaluation Harness:
1. SpeakerSimilarityEvaluator embedding extraction & cosine similarity math
2. Biometric discrimination: Identical speakers score >= 0.95; distinct speakers score < 0.70
3. Likeness tier thresholds (Tier 3 >= 0.85, Tier 2, Tier 1, Divergent)
4. ASR Word Error Rate (WER) & Character Error Rate (CER) calculation via jiwer
5. Case & punctuation invariance in transcription evaluation
"""

import pytest
import numpy as np

from app.ai.audio.similarity import (
    speaker_similarity_evaluator,
    SpeakerSimilarityResult,
    SpeakerSimilarityEvaluator
)
from scripts.evaluate_asr_wer import compute_wer_metrics, normalize_text_for_wer


def generate_tone_speech(freq_hz: float, duration_sec: float = 2.0, sr: int = 16000) -> np.ndarray:
    """Generates synthetic harmonic voiced audio signal for deterministic testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = np.zeros_like(t)
    for h in range(1, 10):
        sig += (0.3 / h) * np.sin(2 * np.pi * freq_hz * h * t)
    # Apply soft envelope
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.0 * t))
    return (sig * env).astype(np.float32)


def test_evaluator_interface():
    """Verify SpeakerSimilarityEvaluator exposes complete contract."""
    evaluator = speaker_similarity_evaluator
    assert isinstance(evaluator, SpeakerSimilarityEvaluator)
    assert hasattr(evaluator, "extract_ecapa_embedding")
    assert hasattr(evaluator, "extract_wavlm_embedding")
    assert hasattr(evaluator, "compute_similarity")
    assert hasattr(evaluator, "compare_files")


def test_identical_speaker_similarity():
    """Verify that identical voice samples achieve >= 0.95 similarity and pass Tier 3."""
    sr = 16000
    voice = generate_tone_speech(130.0, duration_sec=2.5, sr=sr)
    # Slight amplitude variation
    voice_dup = (voice * 0.98).astype(np.float32)

    result = speaker_similarity_evaluator.compute_similarity(voice, voice_dup, sr, sr)

    assert isinstance(result, SpeakerSimilarityResult)
    assert result.ecapa_similarity >= 0.95
    assert result.wavlm_similarity >= 0.95
    assert result.composite_score >= 0.95
    assert result.target_met is True
    assert "Tier 3" in result.likeness_tier


def test_cross_speaker_discrimination():
    """Verify that acoustically divergent speakers score below Tier 3 target threshold."""
    sr = 16000
    voice_male = generate_tone_speech(110.0, duration_sec=2.5, sr=sr)
    voice_female = generate_tone_speech(260.0, duration_sec=2.5, sr=sr)

    result = speaker_similarity_evaluator.compute_similarity(voice_male, voice_female, sr, sr)

    assert isinstance(result, SpeakerSimilarityResult)
    assert result.ecapa_similarity < 0.75
    assert result.target_met is False
    assert result.composite_score < 0.70


def test_wer_calculation_exact_match():
    """Verify identical text yields zero WER/CER and passes accuracy gate."""
    ref = "EchoVoice phase zero evaluation harness"
    hyp = "EchoVoice phase zero evaluation harness"

    res = compute_wer_metrics(reference=ref, hypothesis=hyp)
    assert res.wer == 0.0
    assert res.cer == 0.0
    assert res.accuracy_rate == 1.0
    assert res.substitutions == 0
    assert res.insertions == 0
    assert res.deletions == 0
    assert res.passed is True


def test_wer_calculation_normalization():
    """Verify punctuation and case differences do not penalize WER."""
    ref = "Hello, World! This is Echo-Voice."
    hyp = "hello world this is echo voice"

    res = compute_wer_metrics(reference=ref, hypothesis=hyp)
    assert res.wer == 0.0
    assert res.passed is True


def test_wer_calculation_with_errors():
    """Verify substitutions and insertions are counted accurately."""
    ref = "the quick brown fox"
    hyp = "the fast brown fox jumped"

    res = compute_wer_metrics(reference=ref, hypothesis=hyp)
    # ref: 4 words. hyp: 1 substitution ("fast" for "quick"), 1 insertion ("jumped")
    # WER = (1 sub + 1 ins) / 4 = 2 / 4 = 0.50
    assert res.substitutions == 1
    assert res.insertions == 1
    assert res.wer == 0.50
    assert res.passed is False
