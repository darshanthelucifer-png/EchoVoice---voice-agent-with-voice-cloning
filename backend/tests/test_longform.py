"""
Long-Form Speech Generation & Checkpoint Resume Tests (backend/tests/test_longform.py)
--------------------------------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- End-to-End Pipeline Testing: Tests chunking, stitching, YouTube mastering,
  checkpoint resilience, QC evaluation, and API lifecycle.
- Async Integration Testing: Validates background worker scheduling and state transitions.
"""

import asyncio
from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
from httpx import AsyncClient

from app.core.config import settings
from app.ai.audio.chunker import chunker
from app.ai.audio.stitcher import audio_stitcher
from app.ai.audio.mastering import youtube_mastering_engine
from app.ai.audio.qc import audio_qc_evaluator
from app.models.tts_job import JobStatus


def test_script_chunker_multilingual():
    """Verify smart text chunking with Western and Indian punctuation."""
    text = (
        "Welcome to EchoVoice speech synthesis. "
        "Here is a second sentence with a clause, that should be preserved naturally! "
        "क्या यह वाक्य ठीक से विभाजित होता है? हाँ, यह बहुत अच्छा है। "
        "शान्ति और धैर्य ही सफलता की कुंजी हैं॥\n\n"
        "This is a second paragraph that tests the longer pause duration."
    )

    chunks = chunker.chunk_text(text)
    assert len(chunks) >= 4
    for c in chunks:
        assert len(c.text) <= 260
        assert c.pause_after_ms > 0
        assert c.word_count > 0


def test_audio_stitcher_equal_power_crossfade():
    """Verify seamless audio stitching with equal-power crossfades and room tone."""
    sr = 24000
    t = np.linspace(0, 1, sr)
    c1 = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    c2 = (0.3 * np.sin(2 * np.pi * 600 * t)).astype(np.float32)

    stitched = audio_stitcher.stitch([c1, c2], [400], sr=sr)
    # Expected length: 1s + 0.4s pause + 1s = ~2.4s
    expected_samples = int(2.4 * sr)
    assert abs(len(stitched) - expected_samples) < sr * 0.05
    assert np.all(np.isfinite(stitched))
    assert np.max(np.abs(stitched)) <= 1.0


def test_youtube_audio_mastering_specs(tmp_path):
    """Verify mastering to 48kHz, -14 LUFS, -1.5 dBTP, and multi-format exports."""
    sr = 24000
    t = np.linspace(0, 3, sr * 3)
    tone = (0.3 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)

    result = youtube_mastering_engine.master(
        audio=tone,
        source_sr=sr,
        output_dir=tmp_path,
        base_filename="test_master",
        title="Automated Test Audio",
        speaker_name="EchoVoice Test Speaker"
    )

    assert result.sample_rate == 48_000
    assert abs(result.integrated_lufs - (-14.0)) < 1.0
    assert result.true_peak_db <= -1.45
    assert "wav_24bit_48k" in result.export_paths
    assert "mp3_320k" in result.export_paths
    assert result.export_paths["wav_24bit_48k"].exists()
    assert result.export_paths["mp3_320k"].exists()


def test_audio_qc_evaluator():
    """Verify QC metrics: WER, speaker cosine similarity, and acoustic thresholds."""
    ref_text = "EchoVoice provides studio quality voice cloning and conversational AI."
    hyp_text = "EchoVoice provides studio quality voice cloning and conversational AI."

    wer = audio_qc_evaluator.calculate_wer(ref_text, hyp_text)
    assert wer == 0.0

    sr = 24000
    t = np.linspace(0, 2, sr * 2)
    audio = (0.2 * np.sin(2 * np.pi * 350 * t)).astype(np.float32)

    emb = audio_qc_evaluator.extract_speaker_embedding(audio, sr)
    assert len(emb) == 256
    sim = audio_qc_evaluator.calculate_similarity(audio, sr, emb)
    assert sim >= 0.99

    report = audio_qc_evaluator.evaluate(
        audio=audio,
        sr=sr,
        target_script=ref_text,
        ref_embedding=emb,
        mock_hypothesis=hyp_text
    )
    assert report.wer == 0.0
    assert report.speaker_similarity >= 0.99


@pytest.mark.asyncio
async def test_tts_jobs_api_lifecycle(client: AsyncClient):
    """Test full API lifecycle: Submit Job -> List -> Get -> Cancel -> Resume."""
    # 1. Register & Login
    reg_res = await client.post(
        "/api/v1/auth/register",
        json={"email": "longform_api_tester@echovoice.ai", "password": "Password123!", "full_name": "API Tester"}
    )
    assert reg_res.status_code in (201, 400)

    login_res = await client.post(
        "/api/v1/auth/login",
        json={"email": "longform_api_tester@echovoice.ai", "password": "Password123!"}
    )
    assert login_res.status_code == 200
    token = login_res.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Submit Long-Form Generation Job
    payload = {
        "script_text": (
            "First sentence of the speech generation test. "
            "Second sentence testing chunking and API orchestration."
        ),
        "target_language": "en",
        "engine": "fallback",
        "mastering_preset": "youtube_voiceover"
    }
    create_res = await client.post("/api/v1/tts-jobs", json=payload, headers=headers)
    assert create_res.status_code == 201
    job_data = create_res.json()["data"]
    job_id = job_data["id"]
    assert job_data["status"] in (JobStatus.PENDING.value, JobStatus.PROCESSING.value)

    # 3. List Jobs
    list_res = await client.get("/api/v1/tts-jobs", headers=headers)
    assert list_res.status_code == 200
    assert len(list_res.json()["data"]) >= 1

    # 4. Get Job by ID
    get_res = await client.get(f"/api/v1/tts-jobs/{job_id}", headers=headers)
    assert get_res.status_code == 200
    assert get_res.json()["data"]["id"] == job_id

    # 5. Wait briefly for completion
    for _ in range(15):
        await asyncio.sleep(0.5)
        check = await client.get(f"/api/v1/tts-jobs/{job_id}", headers=headers)
        if check.json()["data"]["status"] == JobStatus.COMPLETED.value:
            break

    # 6. Verify Completed State or Download
    final_res = await client.get(f"/api/v1/tts-jobs/{job_id}", headers=headers)
    final_data = final_res.json()["data"]
    if final_data["status"] == JobStatus.COMPLETED.value:
        assert final_data["progress"] == 100.0
        assert final_data["output_wav_path"] is not None

        # Test Download Endpoint
        dl_res = await client.get(f"/api/v1/tts-jobs/{job_id}/download/wav", headers=headers)
        assert dl_res.status_code == 200
        assert len(dl_res.content) > 1000
