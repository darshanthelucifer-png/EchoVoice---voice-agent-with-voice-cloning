"""
Health & Diagnostic Endpoint Tests (backend/tests/test_health.py)
-----------------------------------------------------------------
Verifies service availability, database connectivity, and configuration reporting.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check_unversioned(client: AsyncClient):
    """Verify /health returns 200 and healthy status."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["app_name"] == "EchoVoice"
    assert data["database_connected"] is True
    assert "configured_models" in data
    assert data["configured_models"]["asr_model"] == "openai/whisper-large-v3-turbo"


@pytest.mark.asyncio
async def test_health_check_versioned(client: AsyncClient):
    """Verify /api/v1/health returns 200."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
