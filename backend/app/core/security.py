"""
Security & Cryptography Module (backend/app/core/security.py)
-------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Direct cryptographic primitives via `bcrypt`: Avoids deprecated wrappers,
  guaranteeing zero runtime errors and fast salt-hashed passwords.
- JWT (JSON Web Tokens) with PyJWT: Stateless authentication with payload
  claims, issuance (`iat`), and expiration (`exp`) timestamps.
- Explicit Type Annotations: Full static typing with `Optional[timedelta]`,
  `str`, and `dict`.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional
import bcrypt
import jwt

from app.core.config import settings


def get_password_hash(password: str) -> str:
    """
    Hashes a plaintext password using bcrypt with automatic salt generation.
    Returns a UTF-8 encoded string suitable for DB storage.
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Constant-time comparison of a plaintext password against a stored bcrypt hash.
    Protects against timing attacks.
    """
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8")
        )
    except Exception:
        return False


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    extra_claims: Optional[Dict[str, Any]] = None
) -> str:
    """
    Encodes a signed JSON Web Token (JWT) containing the subject (user ID) and expiration.
    """
    now = datetime.now(timezone.utc)
    if expires_delta:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    payload: Dict[str, Any] = {
        "sub": str(subject),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "type": "access",
    }
    if extra_claims:
        payload.update(extra_claims)

    encoded_jwt = jwt.encode(
        payload,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    return encoded_jwt


def decode_access_token(token: str) -> Dict[str, Any]:
    """
    Decodes and validates the signature and expiration of a JWT token.
    Raises jwt.PyJWTError if expired, malformed, or tampered with.
    """
    payload = jwt.decode(
        token,
        settings.JWT_SECRET,
        algorithms=[settings.JWT_ALGORITHM]
    )
    return payload
