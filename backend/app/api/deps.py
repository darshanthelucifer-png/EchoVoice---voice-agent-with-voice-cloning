"""
API Dependencies Module (backend/app/api/deps.py)
-------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- FastAPI Dependency Injection (`Depends`): Composes decoupled dependencies
  (database sessions, token parsing, and user authentication) cleanly across route handlers.
- OAuth2 Password Bearer Specification: Enables automatic OpenAPI (Swagger UI)
  integration with the "Authorize" lock button.
- Exception Handling with HTTP 401 Bearer challenges.
"""

from typing import AsyncGenerator
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.user import User
from app.services.auth_service import auth_service

# OAuth2 scheme for Swagger UI documentation
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl=f"{settings.API_V1_STR}/auth/login/oauth",
    auto_error=True
)


async def get_current_user(
    db: AsyncSession = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    """
    Validates the Bearer token, extracts the subject (user_id), and fetches
    the active User record from the database.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except (jwt.PyJWTError, Exception):
        raise credentials_exception

    user = await auth_service.get_by_id(db, user_id)
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )
    return user


async def get_current_active_superuser(
    current_user: User = Depends(get_current_user)
) -> User:
    """Ensures the authenticated user possesses superuser privileges."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user
