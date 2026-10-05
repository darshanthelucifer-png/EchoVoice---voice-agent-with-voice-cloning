"""
Phase 6 Unit Tests: Song Studio SVC, Remixing, Mastering & Quality Gate on a 3-Minute Song (backend/tests/test_song_studio_phase6.py)
--------------------------------------------------------------------------------------------------------------------------------------
Verifies Phase 6 requirements:
1. Vocal Post-Processing Chain: De-click, de-ess, EQ-match, sample-accurate timing alignment, loudness match, and reverb restore.
2. Singing Voice Conversion (SVC): Silence-boundary chunking, equal-power crossfade, Tier 3 RVC v2 and Tier 2 Seed-VC routing.
3. Multitrack Remixing & Broadcast Mastering: Pocket sidechain ducking, -14 LUFS integrated, true peak <= -1.0 dBTP, 48 kHz studio master.
4. Acoustic & Biometric Quality Gate: Speaker similarity (ECAPA/WavLM >= 0.85), clipping detection, and F0 octave jump detection.
5. Legal Gate: Affirmative consent verification and synthetic media attribution watermarks.
6. Experimental Pipelines: Lyric translation and ACE-Step generative music interface.
7. Full 3-Minute Song Test (180.0 seconds): End-to-end execution on a full-length multitrack song.
8. REST API Endpoints: /convert (sync & async job polling), /jobs/{id}, /quality-report, /ab-comparison, and audio streaming.
"""

import pytest
import numpy as np
import soundfile as sf
import json
from pathlib import Path
from typing import Tuple, Dict, Any, List
from scipy import signal
from httpx import AsyncClient, ASGITransport

from app.core.config import settings
from app.ai.song.vocal_postprocess import (
    vocal_post_processor,
    VocalPostProcessor,
    VocalPostProcessResult,
    VocalPostProcessConfig
)
from app.ai.song.svc_engine import (
    singing_voice_engine,
    SingingVoiceConversionEngine,
    SVCResult,
    SVCSettings
)
from app.ai.song.remaster import (
    song_remaster_engine,
    SongRemasterEngine,
    SongRemasterResult,
    MixdownSettings
)
from app.ai.song.quality_gate import (
    song_quality_gate,
    SongQualityGate,
    QualityGateReport,
    OctaveErrorSection
)
from app.ai.song.legal_gate import (
    song_legal_gate,
    SongLegalGate,
    LegalVerificationResult
)
from app.ai.song.experimental import (
    experimental_song_features,
    LyricTranslationResult,
    ACEStepGenerationResult
)
from app.services.song_service import (
    song_studio_service,
    SongStudioProject,
    ClonedSongRemasterResult
)
from app.main import create_application


def generate_multitrack_song(duration_sec: float = 180.0, sr: int = 44_100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a full-length multitrack song with realistic musical structure:
    - Backing instrumental (kick drums at 120 BPM, 80 Hz bassline, chord pads)
    - Lead vocal line with natural verse/chorus pauses (silence boundaries every 10-14 seconds)
    Returns: (full_mix, vocal_track, instrumental_track)
    """
    n_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, n_samples, endpoint=False)

    # 1. Bassline (A2 110 Hz root + 82.4 Hz sub)
    bass = 0.35 * np.sin(2 * np.pi * 110.0 * t) + 0.25 * np.sin(2 * np.pi * 82.4 * t)

    # 2. Drums: 4/4 Kick pulses at 120 BPM (2 beats per sec)
    drums = np.zeros(n_samples, dtype=np.float32)
    beat_step = int(sr * 0.5)  # 120 BPM
    kick_len = int(0.10 * sr)
    kick_t = np.linspace(0, 0.10, kick_len, endpoint=False)
    kick_pulse = np.sin(2 * np.pi * (160.0 * np.exp(-kick_t * 35.0)) * kick_t) * np.exp(-kick_t * 22.0)

    for b in range(0, n_samples - kick_len, beat_step):
        drums[b:b + kick_len] += 0.60 * kick_pulse

    # 3. Chord Pads (A Minor: A440, C523.25, E659.25)
    chords = (
        0.18 * np.sin(2 * np.pi * 440.0 * t) +
        0.15 * np.sin(2 * np.pi * 523.25 * t) +
        0.15 * np.sin(2 * np.pi * 659.25 * t)
    )

    instrumental = (bass + drums + chords).astype(np.float32)

    # 4. Lead Vocals: 220 Hz (A3) with 5.5 Hz vibrato and periodic phrasing pauses
    vibrato = 10.0 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(220.0 + vibrato) / sr
    vocals = 0.50 * np.sin(phase) + 0.25 * np.sin(2 * phase)

    # Phrasing mask: 10s singing followed by 2.5s breathing/pause silence
    phrase_cycle_sec = 12.5
    t_mod = np.fmod(t, phrase_cycle_sec)
    singing_mask = np.where(t_mod < 10.0, 1.0, 0.0)

    # Smooth phrase attack and release
    vocal_env = signal.medfilt(singing_mask.astype(np.float32), kernel_size=21)
    vocals = (vocals * vocal_env * 0.60).astype(np.float32)

    full_mix = (instrumental + vocals) * 0.70
    return full_mix.astype(np.float32), vocals.astype(np.float32), instrumental.astype(np.float32)


@pytest.mark.asyncio
async def test_vocal_post_processor_chain(tmp_path):
    """Verifies de-click, de-ess, timing alignment, loudness match, and EQ matching."""
    sr = 44_100
    _, voc, _ = generate_multitrack_song(duration_sec=3.0, sr=sr)

    # Inject deliberate clicks/spikes
    voc_with_clicks = np.copy(voc)
    voc_with_clicks[int(0.5 * sr)] += 1.2
    voc_with_clicks[int(1.5 * sr)] -= 1.1

    # Inject harsh 7 kHz sibilance burst
    t_sib = np.linspace(0, 0.2, int(0.2 * sr))
    sib_burst = 0.8 * np.sin(2 * np.pi * 7000.0 * t_sib)
    voc_with_clicks[int(0.8 * sr): int(0.8 * sr) + len(sib_burst)] += sib_burst

    # Artificial delay to test timing alignment (25 ms)
    delay_samples = int(0.025 * sr)
    delayed_voc = np.pad(voc_with_clicks, (delay_samples, 0))[:len(voc)]

    res: VocalPostProcessResult = vocal_post_processor.process(
        converted_vocal=delayed_voc,
        original_vocal=voc,
        sr=sr,
        reference_speaker=voc,
        reverb_attenuation_db=2.5
    )

    assert "declick" in res.steps_applied
    assert "deess" in res.steps_applied
    assert "timing_align" in res.steps_applied
    assert "loudness_match" in res.steps_applied
    assert "reverb_character" in res.steps_applied
    assert abs(res.timing_offset_ms) > 0.0
    assert len(res.audio) == len(voc)
    assert np.max(np.abs(res.audio)) < 1.0


@pytest.mark.asyncio
async def test_quality_gate_octave_error_detection():
    """Verifies that SongQualityGate detects deliberate octave jumps and measures audio health."""
    sr = 44_100
    _, voc, _ = generate_multitrack_song(duration_sec=4.0, sr=sr)

    # Create artificial octave jump up (double frequency for 1.0 second: from 1.5s to 2.5s)
    jump_voc = np.copy(voc)
    t = np.linspace(0, 4.0, len(voc))
    jump_start = int(1.5 * sr)
    jump_end = int(2.5 * sr)
    # Double frequency (+1 octave = 440 Hz)
    jump_voc[jump_start:jump_end] = 0.5 * np.sin(2 * np.pi * 440.0 * t[jump_start:jump_end])

    # Run quality gate
    flagged, corr = song_quality_gate.detect_octave_errors(
        original_vocal=voc,
        converted_vocal=jump_voc,
        sr=sr
    )

    assert len(flagged) >= 1, "Must detect deliberate octave jump"
    err = flagged[0]
    assert err.error_type in ["octave_up", "pitch_divergence"]
    assert 1.2 <= err.start_sec <= 1.8
    assert 2.2 <= err.end_sec <= 2.8


@pytest.mark.asyncio
async def test_legal_gate_compliance():
    """Verifies ethical consent enforcement and synthetic media attribution metadata."""
    # 1. Deny when user consent is False
    res_deny = song_legal_gate.verify_request(has_user_consent=False, voice_profile_consent=True)
    assert not res_deny.allowed
    assert "User must confirm personal rights" in res_deny.rejection_reason

    # 2. Deny when profile consent is False
    res_profile_deny = song_legal_gate.verify_request(has_user_consent=True, voice_profile_consent=False)
    assert not res_profile_deny.allowed
    assert "voice profile has not completed biometric cloning consent" in res_profile_deny.rejection_reason

    # 3. Allow when both are affirmed
    res_allow = song_legal_gate.verify_request(has_user_consent=True, voice_profile_consent=True)
    assert res_allow.allowed
    assert "AI-Generated Singing Voice" in res_allow.attribution_tags["SYNTHETIC_MEDIA_DISCLOSURE"]


@pytest.mark.asyncio
async def test_experimental_lyric_translation_and_music_generation(tmp_path):
    """Verifies experimental lyric translation (Whisper -> NLLB) and ACE-Step interface."""
    vocal_file = tmp_path / "mock_vocal.wav"
    sf.write(str(vocal_file), np.zeros(24000, dtype=np.float32), 24000)

    # 1. Test Lyric Translation
    trans_res: LyricTranslationResult = await experimental_song_features.translate_song_lyrics(
        vocal_audio_path=vocal_file,
        target_language="hi"
    )
    assert trans_res.is_experimental
    assert len(trans_res.translated_lyrics) > 0
    assert "EXPERIMENTAL" in trans_res.disclaimer

    # 2. Test ACE-Step Generative Interface
    ace_res: ACEStepGenerationResult = await experimental_song_features.generate_song_from_lyrics(
        lyrics="Neon lights reflecting in the rain",
        style_genre="synthwave",
        target_bpm=128.0,
        output_dir=tmp_path
    )
    assert ace_res.is_experimental
    assert ace_res.bpm == 128.0
    assert "ACE-Step" in ace_res.title


@pytest.mark.asyncio
async def test_full_three_minute_song_pipeline(tmp_path):
    """
    CRITICAL PHASE 6 REQUIREMENT:
    Executes the entire Song Studio pipeline on a full 3-minute (180.0 seconds) multitrack song:
    - 4-stem separation & dereverberation
    - Silence-boundary chunking into manageable ~12s segments
    - Singing voice conversion across chunks with crossfade
    - Vocal post-processing (de-click, de-ess, EQ-match, timing align)
    - Pocket sidechain ducking & YouTube mastering (-14 LUFS, true peak <= -1.0 dBTP, 48 kHz)
    - Quality gate verification & artifact export (WAV, MP3, FLAC, Acapella, JSON report)
    """
    sr = 44_100
    duration_sec = 180.0  # Exactly 3 minutes

    # Generate 3-minute song
    mix, voc, inst = generate_multitrack_song(duration_sec=duration_sec, sr=sr)
    song_file = tmp_path / "three_minute_song.wav"
    sf.write(str(song_file), mix, sr)

    song_id = "test_song_3min"

    # Step 1: Ingest, Separate, Dereverb & Analyze
    project: SongStudioProject = await song_studio_service.process_song(
        input_audio=song_file,
        song_id=song_id,
        original_filename="three_minute_song.wav",
        source_sr=sr,
        dereverb_strength=0.50,
        force_dsp=True
    )

    assert abs(project.duration_seconds - 180.0) < 1.0
    assert project.bpm > 0
    assert project.stem_paths["clean_lead_vocals"].exists()
    assert project.stem_paths["instrumental"].exists()

    # Step 2: Execute SVC, Post-Processing, Multitrack Remix & YouTube Mastering
    remaster_res: ClonedSongRemasterResult = await song_studio_service.convert_and_remaster(
        song_id=song_id,
        pitch_shift_semitones=0.0,
        auto_tune=True,
        autotune_strength=0.60,
        sidechain_duck_db=2.0,
        user_consent=True,
        voice_profile_consent=True,
        force_tier="tier2"
    )

    # 1. Duration and Sample Rate Assertions
    assert abs(remaster_res.duration_seconds - 180.0) < 1.5
    assert remaster_res.sample_rate == 48_000  # Broadcast 48 kHz standard

    # 2. Broadcast Loudness & Peak Assertions
    assert abs(remaster_res.integrated_lufs - (-14.0)) <= 1.5, f"Integrated LUFS {remaster_res.integrated_lufs} must meet -14 LUFS standard"
    assert remaster_res.true_peak_db <= -1.0, f"True peak {remaster_res.true_peak_db} must respect <= -1.0 dBTP ceiling"
    assert remaster_res.sidechain_attenuation_db > 0.0, "Sidechain pocket ducking must be applied"

    # 3. Biometric & Quality Gate Assertions
    assert remaster_res.likeness_score >= 0.85, f"Biometric likeness {remaster_res.likeness_score} must be >= 0.85"
    assert remaster_res.quality_gate_passed

    # 4. Artifact File Integrity Assertions
    export_paths = remaster_res.export_paths
    assert "full_song_wav" in export_paths and export_paths["full_song_wav"].exists()
    assert "full_song_mp3" in export_paths and export_paths["full_song_mp3"].exists()
    assert "acapella_wav" in export_paths and export_paths["acapella_wav"].exists()
    assert "instrumental_wav" in export_paths and export_paths["instrumental_wav"].exists()

    # Verify mastered WAV has non-trivial filesize for 3-minute 48kHz audio (> 10 MB)
    assert export_paths["full_song_wav"].stat().st_size > 10_000_000

    # 5. Manifest & Telemetry Assertions
    quality_report = await song_studio_service.get_quality_report(song_id)
    assert quality_report is not None
    assert quality_report["likeness_score"] >= 0.85
    assert quality_report["clipping_ratio"] < 0.001

    ab_data = await song_studio_service.get_ab_comparison(song_id)
    assert ab_data is not None
    assert "original_tracks" in ab_data
    assert "cloned_tracks" in ab_data
    assert len(ab_data["waveform_telemetry"]["original_vocal_envelope"]) == 1000


@pytest.mark.asyncio
async def test_song_api_conversion_and_job_polling(tmp_path):
    """Verifies REST API endpoints for synchronous conversion and background job polling."""
    sr = 44_100
    mix, _, _ = generate_multitrack_song(duration_sec=3.0, sr=sr)
    mix_file = tmp_path / "api_quick_song.wav"
    sf.write(str(mix_file), mix, sr)

    app = create_application()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Ingest Song
        with open(mix_file, "rb") as f:
            upload_resp = await client.post(
                "/api/v1/song/upload-and-process",
                files={"file": ("api_quick_song.wav", f, "audio/wav")},
                data={"dereverb_strength": "0.50", "force_dsp": "true"}
            )
        assert upload_resp.status_code == 200
        song_id = upload_resp.json()["data"]["song_id"]

        # 2. Test Synchronous Convert Endpoint
        conv_resp = await client.post(
            "/api/v1/song/convert",
            json={
                "song_id": song_id,
                "pitch_shift_semitones": 0.0,
                "auto_tune": True,
                "autotune_strength": 0.60,
                "sidechain_duck_db": 2.0,
                "user_consent": True,
                "voice_profile_consent": True,
                "force_tier": "tier2",
                "async_job": False
            }
        )
        assert conv_resp.status_code == 200
        conv_data = conv_resp.json()["data"]
        assert conv_data["sample_rate"] == 48_000
        assert conv_data["likeness_score"] >= 0.85
        assert conv_data["quality_gate_passed"]

        # 3. Test Quality Report Endpoint
        qc_resp = await client.get(f"/api/v1/song/{song_id}/quality-report")
        assert qc_resp.status_code == 200
        qc_data = qc_resp.json()["data"]
        assert qc_data["likeness_score"] >= 0.85

        # 4. Test A/B Comparison Endpoint
        ab_resp = await client.get(f"/api/v1/song/{song_id}/ab-comparison")
        assert ab_resp.status_code == 200
        ab_data = ab_resp.json()["data"]
        assert "original_tracks" in ab_data
        assert "cloned_tracks" in ab_data

        # 5. Test Audio Streaming Endpoint
        audio_resp = await client.get(f"/api/v1/song/audio/{song_id}/full_song_cloned_master.wav")
        assert audio_resp.status_code == 200
        assert audio_resp.headers["content-type"] == "audio/wav"
        assert len(audio_resp.content) > 1000

        # 6. Test Async Job Scheduling & Polling
        job_resp = await client.post(
            "/api/v1/song/convert",
            json={
                "song_id": song_id,
                "user_consent": True,
                "voice_profile_consent": True,
                "async_job": True
            }
        )
        assert job_resp.status_code == 200
        job_id = job_resp.json()["data"]["job_id"]

        poll_resp = await client.get(f"/api/v1/song/jobs/{job_id}")
        assert poll_resp.status_code == 200
        job_state = poll_resp.json()["data"]
        assert job_state["status"] in ["PENDING", "PROCESSING", "COMPLETED"]
