"""
Authentication Schemas (backend/app/schemas/auth.py)
----------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic models for OAuth2 Bearer token representation and JWT decoding payload.
"""

from typing import Optional
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """Returned upon successful login/registration."""
    access_token: str
    token_type: str = "bearer"
    expires_in_seconds: int


class TokenPayload(BaseModel):
    """Payload decoded from valid JWT."""
    sub: Optional[str] = None
    exp: Optional[int] = None
    type: Optional[str] = None


class LoginRequest(BaseModel):
    """Standard JSON login payload."""
    email: EmailStr
    password: str
