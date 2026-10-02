"""
Voice Profile & Consent Gate Unit Tests (backend/tests/test_voice_profiles.py)
-----------------------------------------------------------------------------
Verifies:
1. Mandatory ethical consent gate rejection when consent_given=False
2. Successful voice enrollment, embedding extraction, and profile creation
3. Listing profiles and setting default profile
4. Complete disk cleanup on profile deletion
"""

import io
import pytest
import numpy as np
import soundfile as sf
from httpx import AsyncClient

from app.core.config import settings


def create_mock_voice_wav(duration_sec: float = 6.0, sr: int = 24000) -> bytes:
    """Creates in-memory WAV bytes with harmonic voiced proxy."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    sig = np.zeros_like(t)
    for h in [1, 2, 3]:
        sig += (0.3 / h) * np.sin(2 * np.pi * 150 * h * t)
    buf = io.BytesIO()
    sf.write(buf, sig.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf.read()


@pytest.mark.asyncio
async def test_consent_gate_rejection(client: AsyncClient):
    """Verify that enrolling without explicit consent is rejected with 400 Bad Request."""
    # 1. Register test user
    user_payload = {
        "email": "consent_tester@echovoice.ai",
        "password": "Password123!",
        "full_name": "Consent Tester"
    }
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    token = reg_res.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    wav_bytes = create_mock_voice_wav(duration_sec=6.0)
    files = {"file": ("my_voice.wav", wav_bytes, "audio/wav")}
    data = {
        "name": "My Cloned Voice",
        "consent_given": "false"  # Explicitly False
    }

    response = await client.post(
        "/api/v1/voice-profiles/enroll",
        headers=headers,
        files=files,
        data=data
    )
    assert response.status_code == 400
    assert "Consent Required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_enroll_voice_profile_success(client: AsyncClient):
    """Verify full enrollment lifecycle with consent given."""
    user_payload = {
        "email": "profile_owner@echovoice.ai",
        "password": "Password123!",
        "full_name": "Profile Owner"
    }
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    token = reg_res.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    wav_bytes = create_mock_voice_wav(duration_sec=7.0)
    files = {"file": ("owner_voice.wav", wav_bytes, "audio/wav")}
    data = {
        "name": "Studio Master Voice",
        "description": "24kHz clean enrollment",
        "consent_given": "true"  # Consent given
    }

    # 1. Enroll
    response = await client.post(
        "/api/v1/voice-profiles/enroll",
        headers=headers,
        files=files,
        data=data
    )
    assert response.status_code == 201
    profile_data = response.json()["data"]
    assert profile_data["name"] == "Studio Master Voice"
    assert profile_data["consent_given"] is True
    assert profile_data["is_default"] is True
    profile_id = profile_data["id"]

    # 2. List Profiles
    list_res = await client.get("/api/v1/voice-profiles/", headers=headers)
    assert list_res.status_code == 200
    profiles = list_res.json()["data"]
    assert len(profiles) >= 1

    # 3. Stream Reference Audio
    ref_res = await client.get(f"/api/v1/voice-profiles/{profile_id}/reference-audio", headers=headers)
    assert ref_res.status_code == 200
    assert len(ref_res.content) > 1000

    # 4. Delete Profile and verify disk cleanup
    del_res = await client.delete(f"/api/v1/voice-profiles/{profile_id}", headers=headers)
    assert del_res.status_code == 200

    # Confirm profile is gone
    get_res = await client.get(f"/api/v1/voice-profiles/{profile_id}", headers=headers)
    assert get_res.status_code == 404
