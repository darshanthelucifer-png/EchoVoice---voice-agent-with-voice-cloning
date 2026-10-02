"""
Pytest Test Fixtures (backend/tests/conftest.py)
------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pytest Fixtures (`@pytest_asyncio.fixture`): Reusable test fixtures providing
  isolated async test database sessions and an HTTP client.
- Dependency Overrides: FastAPI's `app.dependency_overrides` swaps production
  database sessions for in-memory SQLite (`sqlite+aiosqlite:///:memory:`)
  during test execution.
"""

import sys
from pathlib import Path
from typing import AsyncGenerator
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession

# Ensure backend directory is in sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.database import Base, get_db
from app.main import app

# Isolated In-Memory SQLite Test Engine
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False}
)

TestingSessionLocal = async_sessionmaker(
    bind=test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


@pytest_asyncio.fixture(scope="function", autouse=True)
async def prepare_database():
    """Create all database tables before each test function and drop after."""
    import app.models  # noqa: F401
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Provides an isolated database session per test function."""
    async with TestingSessionLocal() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """
    HTTP AsyncClient fixture wired with the dependency override for get_db.
    """
    async def override_get_db() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac

    app.dependency_overrides.clear()
