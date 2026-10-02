"""
Database Module (backend/app/core/database.py)
----------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Async Generators (`yield` in async function): Provides database sessions via
  FastAPI's dependency injection system, automatically committing or rolling back,
  and closing the session cleanly when the request completes.
- SQLAlchemy 2.0 Declarative Mapping: Modern type-annotated ORM declarative base.
- Connection Pooling & Async Engine: Manages asynchronous database connections
  efficiently across the application lifecycle.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.core.config import settings
from app.core.logging import logger

# SQLAlchemy 2.0 Base Class
Base = declarative_base()

# Configure engine arguments based on DB dialect (SQLite vs PostgreSQL)
engine_kwargs = {"echo": False}
if "sqlite" in settings.DATABASE_URL:
    # check_same_thread=False is required for SQLite when accessed across async tasks
    engine_kwargs["connect_args"] = {"check_same_thread": False}

# Asynchronous engine
engine = create_async_engine(
    settings.DATABASE_URL,
    **engine_kwargs
)

# Async session factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency yielding an asynchronous SQLAlchemy database session.
    
    Python Concept: Async Generator (`yield`)
    - The code before `yield` runs before the route handler (session setup).
    - The `yield` hands the session to the route handler.
    - The code after `yield` (or in finally block) runs after the route handler
      returns (session cleanup/close), ensuring no leaked connections.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """
    Initializes database tables by inspecting SQLAlchemy metadata.
    Called once during FastAPI startup event / lifespan.
    """
    # Import all models here so that SQLAlchemy's Base metadata knows about all tables
    import app.models  # noqa: F401

    logger.info("Initializing database schema...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database schema initialized successfully.")
