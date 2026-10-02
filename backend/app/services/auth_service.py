"""
Authentication Service (backend/app/services/auth_service.py)
-------------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Service Layer Pattern: Decouples business logic and database queries completely
  from API route handlers.
- Async/Await Coroutines: Non-blocking database transactions.
- Exception Handling with HTTP Status semantics: Translates domain errors (duplicate
  email, bad credentials) cleanly for API consumption.
"""

from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from fastapi import HTTPException, status

from app.models.user import User
from app.schemas.user import UserCreate
from app.schemas.auth import Token
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.config import settings
from app.services.base import BaseService


class AuthService(BaseService[User]):
    """
    Manages user registration, credential verification, and JWT issuance.
    """
    def __init__(self):
        super().__init__(User)

    async def get_by_email(self, db: AsyncSession, email: str) -> Optional[User]:
        """Fetch user by lowercase email."""
        result = await db.execute(
            select(User).where(User.email == email.lower().strip())
        )
        return result.scalars().first()

    async def register_user(self, db: AsyncSession, user_in: UserCreate) -> User:
        """
        Registers a new user after verifying that the email is not already taken.
        """
        existing_user = await self.get_by_email(db, user_in.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A user with this email already exists."
            )

        user = User(
            email=user_in.email.lower().strip(),
            hashed_password=get_password_hash(user_in.password),
            full_name=user_in.full_name,
            is_active=True,
            is_superuser=False
        )
        db.add(user)
        await db.flush()
        await db.refresh(user)
        return user

    async def authenticate_user(
        self,
        db: AsyncSession,
        email: str,
        password: str
    ) -> User:
        """
        Validates user credentials. Raises 401 on mismatch or disabled account.
        """
        user = await self.get_by_email(db, email)
        if not user or not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="This user account is inactive."
            )
        return user

    def create_user_token(self, user: User) -> Token:
        """
        Generates signed JWT token for the authenticated user.
        """
        access_token = create_access_token(
            subject=user.id,
            extra_claims={"email": user.email, "is_superuser": user.is_superuser}
        )
        return Token(
            access_token=access_token,
            token_type="bearer",
            expires_in_seconds=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )


auth_service = AuthService()
