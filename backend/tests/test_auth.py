"""
Authentication System Tests (backend/tests/test_auth.py)
--------------------------------------------------------
Verifies user registration, password hashing, JWT creation/validation,
and protected route access controls.
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_register_and_login_flow(client: AsyncClient):
    """
    Tests complete lifecycle:
    1. Register a new user
    2. Login via JSON body
    3. Access protected /me endpoint
    """
    user_payload = {
        "email": "testuser@echovoice.ai",
        "password": "StrongPassword123!",
        "full_name": "EchoVoice Tester"
    }

    # 1. Register
    reg_response = await client.post("/api/v1/auth/register", json=user_payload)
    assert reg_response.status_code == 201
    reg_json = reg_response.json()
    assert reg_json["success"] is True
    assert "access_token" in reg_json["data"]
    token = reg_json["data"]["access_token"]

    # 2. Duplicate Registration Rejection
    dup_response = await client.post("/api/v1/auth/register", json=user_payload)
    assert dup_response.status_code == 400
    assert "already exists" in dup_response.json()["detail"]

    # 3. JSON Login
    login_response = await client.post("/api/v1/auth/login", json={
        "email": "testuser@echovoice.ai",
        "password": "StrongPassword123!"
    })
    assert login_response.status_code == 200
    login_json = login_response.json()
    assert login_json["success"] is True
    assert "access_token" in login_json["data"]

    # 4. Failed Login (Wrong password)
    bad_login = await client.post("/api/v1/auth/login", json={
        "email": "testuser@echovoice.ai",
        "password": "WrongPassword!"
    })
    assert bad_login.status_code == 401

    # 5. Access Protected /me with Bearer Token
    headers = {"Authorization": f"Bearer {token}"}
    me_response = await client.get("/api/v1/auth/me", headers=headers)
    assert me_response.status_code == 200
    me_json = me_response.json()
    assert me_json["success"] is True
    assert me_json["data"]["email"] == "testuser@echovoice.ai"
    assert me_json["data"]["full_name"] == "EchoVoice Tester"
    assert me_json["data"]["is_active"] is True


@pytest.mark.asyncio
async def test_oauth2_form_login(client: AsyncClient):
    """
    Verifies OAuth2 standard form login endpoint (/auth/login/oauth)
    used by Swagger UI documentation.
    """
    user_payload = {
        "email": "oauthuser@echovoice.ai",
        "password": "OAuthPassword123!",
        "full_name": "OAuth User"
    }
    await client.post("/api/v1/auth/register", json=user_payload)

    # Form urlencoded request
    response = await client.post(
        "/api/v1/auth/login/oauth",
        data={
            "username": "oauthuser@echovoice.ai",
            "password": "OAuthPassword123!"
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"


@pytest.mark.asyncio
async def test_protected_route_without_token(client: AsyncClient):
    """Verifies that accessing protected endpoints without authentication raises 401."""
    response = await client.get("/api/v1/auth/me")
    assert response.status_code == 401
