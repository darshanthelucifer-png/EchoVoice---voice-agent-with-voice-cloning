"""
Authentication Router (backend/app/api/routes/auth.py)
-------------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Thin Route Handlers: Handlers act solely as request adapters, delegating all
  validation and database persistence to `auth_service`.
- Multiple Authentication Adapters: Supports both modern JSON payload login
  and standard OAuth2 Form (`OAuth2PasswordRequestForm`) for Swagger UI testing.
- Rate Limiting Decorators: `@limiter.limit` protects brute-force targets.
"""

from fastapi import APIRouter, Depends, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_db, get_current_user
from app.core.rate_limit import limiter
from app.core.config import settings
from app.models.user import User
from app.schemas.user import UserCreate, UserRead
from app.schemas.auth import Token, LoginRequest
from app.schemas.common import APIResponse
from app.services.auth_service import auth_service

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=APIResponse[Token],
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user"
)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def register(
    request: Request,
    user_in: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> APIResponse[Token]:
    """
    Registers a new user account with hashed password and returns an access token.
    """
    user = await auth_service.register_user(db, user_in)
    token = auth_service.create_user_token(user)
    return APIResponse(
        success=True,
        message="User registered successfully",
        data=token
    )


@router.post(
    "/login",
    response_model=APIResponse[Token],
    summary="User login with JSON body"
)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login_json(
    request: Request,
    login_data: LoginRequest,
    db: AsyncSession = Depends(get_db)
) -> APIResponse[Token]:
    """
    Authenticates user with JSON credentials and issues a JWT token.
    """
    user = await auth_service.authenticate_user(
        db,
        email=login_data.email,
        password=login_data.password
    )
    token = auth_service.create_user_token(user)
    return APIResponse(
        success=True,
        message="Authentication successful",
        data=token
    )


@router.post(
    "/login/oauth",
    response_model=Token,
    summary="OAuth2 compatible token login (for Swagger UI authorize button)"
)
@limiter.limit(settings.RATE_LIMIT_AUTH)
async def login_oauth(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db)
) -> Token:
    """
    OAuth2 standard password flow endpoint for Swagger UI "Authorize" lock button.
    """
    user = await auth_service.authenticate_user(
        db,
        email=form_data.username,
        password=form_data.password
    )
    return auth_service.create_user_token(user)


@router.get(
    "/me",
    response_model=APIResponse[UserRead],
    summary="Get current user profile"
)
async def get_me(
    current_user: User = Depends(get_current_user)
) -> APIResponse[UserRead]:
    """
    Returns the profile of the currently authenticated user.
    """
    return APIResponse(
        success=True,
        message="Current user profile fetched",
        data=UserRead.model_validate(current_user)
    )
