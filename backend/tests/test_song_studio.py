"""
Phase 5 Unit Tests: Song Studio Stem Separation, Dereverberation & Musical Analysis (backend/tests/test_song_studio.py)
-----------------------------------------------------------------------------------------------------------------------
Verifies Phase 5 requirements:
1. SOTA 4-stem separation (vocals, drums, bass, other) + coherent instrumental mixdown.
2. Lead vocal dereverberation & room bleed suppression.
3. Continuous F0 pitch contour trajectory extraction and vocal range classification.
4. Musical tempo (BPM) and tonal key/scale detection (Krumhansl-Schmuckler chromagram).
5. Output artifacts: clean_lead_vocals.wav, instrumental.wav, vocal_f0_contour.npy, song_metadata.json.
6. Song Studio REST API endpoints & audio streaming.
"""

import pytest
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
from httpx import AsyncClient, ASGITransport

from app.core.config import settings
from app.ai.song.separator import (
    StemSeparationEngine,
    SeparationResult,
    SeparationConfig,
    stem_separation_engine
)
from app.ai.song.dereverb import (
    VocalDereverberator,
    DereverbResult,
    DereverbConfig,
    vocal_dereverberator
)
from app.ai.song.analyzer import (
    SongMusicalAnalyzer,
    SongAnalysisResult,
    F0ContourResult,
    song_musical_analyzer
)
from app.services.song_service import song_studio_service, SongStudioProject
from app.main import create_application


def synthesize_synthetic_song(duration_sec: float = 4.0, sr: int = 44_100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Synthesizes a multitrack musical snippet:
    - Backing Instrumental: 4/4 Kick drum pulses + 80 Hz bassline + C-major/A-minor triad pads
    - Lead Vocals: A3 (220 Hz) melodic vocal line with vibrato
    Returns: (full_mix, vocal_track, instrumental_track)
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Bass (110 Hz A2 root + 80 Hz sub)
    bass = 0.45 * np.sin(2 * np.pi * 110.0 * t) + 0.3 * np.sin(2 * np.pi * 82.4 * t)

    # 2. Drums (4-on-the-floor kick bursts at 120 BPM = 2 beats per sec)
    drums = np.zeros(n_samples, dtype=np.float32)
    beat_interval = int(sr * 0.5)  # 120 BPM
    for b in range(0, n_samples, beat_interval):
        kick_dur = int(0.12 * sr)
        if b + kick_dur < n_samples:
            kick_t = np.linspace(0, 0.12, kick_dur, endpoint=False)
            sweep = np.sin(2 * np.pi * (150.0 * np.exp(-kick_t * 30.0)) * kick_t)
            drums[b: b + kick_dur] += 0.65 * sweep * np.exp(-kick_t * 20.0)

    # 3. Other/Chords (A Minor chord: A440, C523, E659)
    chords = (
        0.20 * np.sin(2 * np.pi * 440.0 * t) +
        0.18 * np.sin(2 * np.pi * 523.25 * t) +
        0.18 * np.sin(2 * np.pi * 659.25 * t)
    )

    instrumental = (bass + drums + chords).astype(np.float32)

    # 4. Lead Vocals: 220 Hz (A3) with vibrato (5 Hz rate, 12 Hz depth)
    vibrato = 12.0 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(220.0 + vibrato) / sr
    vocals = np.zeros(n_samples, dtype=np.float32)
    for h in [1, 2, 3]:
        vocals += (0.4 / h) * np.sin(h * phase)
    # Natural cadence envelope
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 1.0 * t))
    vocals = (vocals * env * 0.55).astype(np.float32)

    full_mix = (instrumental + vocals) * 0.70
    return full_mix.astype(np.float32), vocals.astype(np.float32), instrumental.astype(np.float32)


@pytest.mark.asyncio
async def test_stem_separation_engine(tmp_path):
    """Verify stem separator isolates 4 stems and builds coherent instrumental mix."""
    sr = 44_100
    mix, voc, inst = synthesize_synthetic_song(duration_sec=3.0, sr=sr)
    mix_file = tmp_path / "test_song.wav"
    sf.write(str(mix_file), mix, sr)

    res: SeparationResult = await stem_separation_engine.separate_stems(
        audio_input=mix_file,
        output_dir=tmp_path,
        song_id="unit_test_song",
        force_dsp=True
    )

    # Assert 5 stems exist
    assert "vocals" in res.stems
    assert "drums" in res.stems
    assert "bass" in res.stems
    assert "other" in res.stems
    assert "instrumental" in res.stems

    # Assert stem files saved
    for stem_name in ["vocals", "drums", "bass", "other", "instrumental", "clean_lead_vocals"]:
        assert stem_name in res.stem_paths
        assert res.stem_paths[stem_name].exists()
        assert res.stem_paths[stem_name].stat().st_size > 1000

    # Assert sample rate and duration
    assert res.sample_rate == sr
    assert abs(res.duration_seconds - 3.0) < 0.2
    assert len(res.stems["instrumental"]) == len(res.stems["vocals"])


@pytest.mark.asyncio
async def test_vocal_dereverberation(tmp_path):
    """Verify dereverberator suppresses late reverberant tails and room bleed."""
    sr = 44_100
    _, voc, _ = synthesize_synthetic_song(duration_sec=3.0, sr=sr)

    # Simulate artificial room reverb impulse response (exponential decay)
    ir_len = int(0.6 * sr)
    ir_t = np.linspace(0, 0.6, ir_len)
    ir = np.exp(-ir_t * 6.0) * (np.random.rand(ir_len) * 2.0 - 1.0)
    ir = (ir / np.max(np.abs(ir)) * 0.35).astype(np.float32)

    # Convolve vocal with room reverb
    reverberant_voc = np.convolve(voc, ir)[:len(voc)] + voc * 0.75
    rev_file = tmp_path / "reverberant_vocal.wav"
    sf.write(str(rev_file), reverberant_voc, sr)

    clean_file = tmp_path / "clean_vocal.wav"
    derev_res = await vocal_dereverberator.process_async(
        audio_input=rev_file,
        sr=sr,
        output_path=clean_file,
        config=DereverbConfig(strength=0.60)
    )

    assert derev_res.duration_seconds > 2.0
    assert derev_res.reverb_attenuation_db > 0.0, "Dereverberator must achieve positive reverb attenuation"
    assert clean_file.exists()
    assert clean_file.stat().st_size > 1000


@pytest.mark.asyncio
async def test_musical_analysis_key_bpm_and_f0(tmp_path):
    """Verify musical analyzer extracts BPM, Key, Scale notes, and continuous F0 contour."""
    sr = 44_100
    mix, voc, inst = synthesize_synthetic_song(duration_sec=4.0, sr=sr)

    res: SongAnalysisResult = await song_musical_analyzer.analyze_song(
        vocal_audio=voc,
        instrumental_audio=inst,
        sr=sr,
        output_dir=tmp_path,
        song_id="test_analysis"
    )

    # 1. BPM assertions (synthetic track has beats at 120 BPM)
    assert 100.0 <= res.bpm <= 140.0, f"Expected BPM ~120, got {res.bpm}"

    # 2. Key & Scale assertions (synthetic track is built on A minor)
    assert res.tonic in ["A", "C", "D", "E", "G"]
    assert res.scale_type in ["minor", "major"]
    assert len(res.scale_notes) == 7
    assert len(res.scale_frequencies_hz) > 10

    # 3. F0 pitch contour assertions
    assert res.f0_contour_path.exists()
    assert res.metadata_json_path.exists()
    assert res.f0_stats["voiced_percentage"] > 20.0
    assert 180.0 <= res.f0_stats["median_f0_hz"] <= 260.0  # Synthetic voice ~220 Hz
    assert res.f0_stats["vocal_register"] in ["Tenor", "Baritone", "Alto / Mezzo-Soprano"]

    # Verify npy file loadable
    loaded_f0 = np.load(str(res.f0_contour_path))
    assert len(loaded_f0) > 50


@pytest.mark.asyncio
async def test_song_studio_service_end_to_end(tmp_path):
    """Verify high-level SongStudioService executes full pipeline and writes all Phase 5 artifacts."""
    sr = 44_100
    mix, _, _ = synthesize_synthetic_song(duration_sec=3.5, sr=sr)
    mix_file = tmp_path / "original_song.wav"
    sf.write(str(mix_file), mix, sr)

    project: SongStudioProject = await song_studio_service.process_song(
        input_audio=mix_file,
        song_id="unit_song_proj",
        original_filename="original_song.wav",
        source_sr=sr,
        dereverb_strength=0.50,
        force_dsp=True
    )

    # 1. Artifact path assertions
    paths = project.stem_paths
    assert paths["clean_lead_vocals"].exists()
    assert paths["instrumental"].exists()
    assert paths["drums"].exists()
    assert paths["bass"].exists()
    assert paths["other"].exists()
    assert paths["f0_contour"].exists()
    assert paths["metadata_json"].exists()

    # 2. Download URL assertions
    urls = project.download_urls
    assert "clean_lead_vocals" in urls
    assert "instrumental" in urls
    assert urls["clean_lead_vocals"].startswith("/api/v1/song/audio/unit_song_proj/")

    # 3. Musical metadata assertions
    assert project.bpm > 0
    assert len(project.musical_key) > 0
    assert project.reverb_attenuation_db >= 0.0


@pytest.mark.asyncio
async def test_song_api_endpoints(tmp_path):
    """Verify Song Studio REST API endpoints for upload, retrieval, and audio streaming."""
    sr = 44_100
    mix, _, _ = synthesize_synthetic_song(duration_sec=2.5, sr=sr)
    mix_file = tmp_path / "api_test_song.wav"
    sf.write(str(mix_file), mix, sr)

    app = create_application()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Test Upload and Process
        with open(mix_file, "rb") as f_audio:
            resp = await client.post(
                "/api/v1/song/upload-and-process",
                files={"file": ("api_test_song.wav", f_audio, "audio/wav")},
                data={"dereverb_strength": "0.50", "force_dsp": "true"}
            )

        assert resp.status_code == 200
        data = resp.json()["data"]
        song_id = data["song_id"]
        assert "clean_lead_vocals" in data["download_urls"]
        assert "instrumental" in data["download_urls"]
        assert data["bpm"] > 0
        assert len(data["musical_key"]) > 0

        # 2. Test Get Song Project
        proj_resp = await client.get(f"/api/v1/song/{song_id}")
        assert proj_resp.status_code == 200
        proj_data = proj_resp.json()["data"]
        assert proj_data["song_id"] == song_id

        # 3. Test Stream Stem Audio
        stream_resp = await client.get(f"/api/v1/song/audio/{song_id}/clean_lead_vocals.wav")
        assert stream_resp.status_code == 200
        assert stream_resp.headers["content-type"] == "audio/wav"
        assert len(stream_resp.content) > 1000
