"""
Phase 3 Unit Tests: Tier 3 RVC v2 Training & Inference (backend/tests/test_tier3_rvc.py)
---------------------------------------------------------------------------------------
Verifies Phase 3 requirements:
1. Audio dataset preprocessing and 3-10s chunk slicing
2. ContentVec / HuBERT feature extraction & Faiss vector index creation
3. RVC v2 `.pth` model checkpoint compilation and state dict integrity
4. `RVCInference` engine feature retrieval and audio synthesis
5. `VoiceEngineOrchestrator` automatic Priority Gate enforcement (Tier 3 priority)
6. Model detection and readiness inspection
"""

import pytest
import numpy as np
import soundfile as sf
import torch
import faiss
from pathlib import Path

from app.core.config import settings
from app.ai.vc.rvc_engine import rvc_engine, RVCInference
from app.services.voice_engine_service import voice_engine_service, TieredSynthesisResult
from scripts.train_rvc import (
    preprocess_training_audio,
    extract_features_from_chunks,
    build_faiss_index,
    train_rvc_checkpoint,
    train_rvc_model
)


def synthesize_test_speech(freq: float = 140.0, dur: float = 12.0, sr: int = 24000) -> np.ndarray:
    """Helper to synthesize multi-second voiced speech sample with natural cadence."""
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    sig = np.zeros_like(t)
    for h in [1, 2, 3, 5]:
        sig += (0.3 / h) * np.sin(2 * np.pi * freq * h * t)
    # Continuous natural speech-like cadence
    cadence = 0.7 + 0.3 * np.sin(2 * np.pi * 0.5 * t)
    return (sig * cadence * 0.4).astype(np.float32)


def test_rvc_preprocessing_and_slicing(tmp_path):
    """Verify raw speech is cleaned, VAD-trimmed, and sliced into 3-10s dataset chunks."""
    sr = 24000
    audio = synthesize_test_speech(freq=135.0, dur=15.0, sr=sr)
    dataset_dir = tmp_path / "dataset"

    chunks = preprocess_training_audio(audio, sr, dataset_dir, min_chunk_sec=3.0, max_chunk_sec=6.0)

    assert len(chunks) >= 2
    for c in chunks:
        assert c.exists()
        arr, c_sr = sf.read(str(c))
        assert c_sr == sr
        dur = len(arr) / c_sr
        assert 2.5 <= dur <= 6.5
        # Ensure peak normalized
        assert np.max(np.abs(arr)) <= 0.86


def test_faiss_index_construction_and_retrieval(tmp_path):
    """Verify feature extraction and exact inner-product Faiss index search."""
    sr = 24000
    audio = synthesize_test_speech(freq=145.0, dur=10.0, sr=sr)
    dataset_dir = tmp_path / "dataset"
    chunks = preprocess_training_audio(audio, sr, dataset_dir)

    features = extract_features_from_chunks(chunks, target_sr=sr)
    assert isinstance(features, np.ndarray)
    assert features.shape[1] == 256
    assert features.shape[0] > 50

    # Build Faiss index
    index_path = tmp_path / "test.index"
    index = build_faiss_index(features, index_path)

    assert index_path.exists()
    assert index.ntotal == features.shape[0]

    # Test retrieval query
    query = features[:5]
    distances, indices = index.search(query, k=4)
    assert distances.shape == (5, 4)
    assert indices.shape == (5, 4)
    # Self-match distance should be approx 1.0 (cosine likeness)
    assert distances[0, 0] >= 0.98


def test_rvc_checkpoint_serialization(tmp_path):
    """Verify compilation and serialization of .pth model weights checkpoint."""
    pth_path = tmp_path / "test_model.pth"
    features = np.random.randn(100, 256).astype(np.float32)
    chunk_paths = [tmp_path / "c1.wav", tmp_path / "c2.wav"]

    ckpt = train_rvc_checkpoint("test_profile", features, chunk_paths, pth_path)

    assert pth_path.exists()
    loaded = torch.load(pth_path, map_location="cpu", weights_only=False)
    assert loaded["profile_id"] == "test_profile"
    assert loaded["version"] == "v2"
    assert "mean_timbre" in loaded
    assert loaded["feature_dim"] == 256


@pytest.mark.asyncio
async def test_rvc_inference_engine_conversion(tmp_path):
    """Verify RVCInference engine executes conversion using model weights and Faiss index."""
    sr = 24000
    profile_id = "test_rvc_inference_profile"
    models_dir = tmp_path / "models"
    p_dir = models_dir / profile_id
    p_dir.mkdir(parents=True, exist_ok=True)

    # Train mini model
    audio = synthesize_test_speech(freq=130.0, dur=8.0, sr=sr)
    dataset_dir = p_dir / "dataset"
    chunks = preprocess_training_audio(audio, sr, dataset_dir)
    features = extract_features_from_chunks(chunks, target_sr=sr)
    build_faiss_index(features, p_dir / f"{profile_id}.index")
    train_rvc_checkpoint(profile_id, features, chunks, p_dir / f"{profile_id}.pth", target_sr=sr)

    # Create dedicated engine pointing to test directory
    custom_rvc = RVCInference(models_dir=models_dir)
    assert custom_rvc.has_profile_model(profile_id) is True

    # Convert voice
    source = synthesize_test_speech(freq=210.0, dur=3.0, sr=sr)
    out = await custom_rvc.convert_voice(
        source_audio=source,
        target_reference=audio,
        profile_id=profile_id,
        source_sr=sr,
        target_sr=sr
    )

    assert out.engine_name == "rvc_v2"
    assert len(out.audio) > 0
    assert out.duration_seconds > 2.0
    assert np.max(np.abs(out.audio)) <= 0.86
    assert out.metadata["has_trained_model"] is True


@pytest.mark.asyncio
async def test_orchestrator_tier3_priority_gate(tmp_path, monkeypatch):
    """Verify VoiceEngineOrchestrator automatically selects Tier 3 when model exists."""
    sr = 24000
    profile_id = "priority_test_speaker"
    models_dir = tmp_path / "models"
    p_dir = models_dir / profile_id
    p_dir.mkdir(parents=True, exist_ok=True)

    # Setup model and index in test models directory
    audio = synthesize_test_speech(freq=130.0, dur=6.0, sr=sr)
    dataset_dir = p_dir / "dataset"
    chunks = preprocess_training_audio(audio, sr, dataset_dir)
    features = extract_features_from_chunks(chunks, target_sr=sr)
    build_faiss_index(features, p_dir / f"{profile_id}.index")
    train_rvc_checkpoint(profile_id, features, chunks, p_dir / f"{profile_id}.pth", target_sr=sr)

    # Redirect global rvc_engine to test directory
    monkeypatch.setattr(rvc_engine, "models_dir", models_dir)
    assert rvc_engine.has_profile_model(profile_id) is True

    # Call orchestrator with profile_id
    result: TieredSynthesisResult = await voice_engine_service.synthesize(
        text="EchoVoice Tier 3 Priority Gate unit test.",
        speaker_wav=audio,
        profile_id=profile_id,
        language="en",
        base_engine="fallback",
        auto_tier_selection=True
    )

    # Tier 3 MUST be prioritized
    assert result.tier_used == "tier3_rvc_v2"
    assert result.final_likeness is not None
    assert result.passed_gate == (result.final_likeness >= settings.TIER3_LIKENESS_GATE)
    assert len(result.audio) > 0
    assert result.telemetry["stage"] == "tier3_rvc_conversion"
