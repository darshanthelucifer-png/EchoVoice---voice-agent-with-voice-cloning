"""
Base Service Module (backend/app/services/base.py)
--------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Generic Types (`TypeVar`, `Generic`): Abstract CRUD service that can operate
  over any SQLAlchemy model type while preserving full IDE type-safety and autocompletion.
- Asynchronous DB Operations (`async/await`): Uses `await session.execute(...)` with
  modern SQLAlchemy 2.0 select statements.
"""

from typing import Generic, TypeVar, Type, Optional, List, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.core.database import Base

ModelType = TypeVar("ModelType", bound=Base)


class BaseService(Generic[ModelType]):
    """
    Generic Base Service demonstrating reusable async CRUD operations.
    """
    def __init__(self, model: Type[ModelType]):
        self.model = model

    async def get_by_id(self, db: AsyncSession, id: Any) -> Optional[ModelType]:
        """Fetch a single record by primary key."""
        result = await db.execute(select(self.model).where(self.model.id == id))
        return result.scalars().first()

    async def get_multi(
        self,
        db: AsyncSession,
        skip: int = 0,
        limit: int = 100
    ) -> List[ModelType]:
        """Fetch multiple records with pagination."""
        result = await db.execute(
            select(self.model).offset(skip).limit(limit)
        )
        return list(result.scalars().all())

    async def delete(self, db: AsyncSession, id: Any) -> bool:
        """Delete a record by primary key."""
        obj = await self.get_by_id(db, id)
        if obj:
            await db.delete(obj)
            await db.flush()
            return True
        return False
