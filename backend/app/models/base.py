"""
Base Model Module (backend/app/models/base.py)
----------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Mixin Classes: Reusable base class defining common primary keys and timestamps
  for all database tables without duplicate code.
- UUIDv4 Primary Keys: Universally Unique Identifiers prevent enumeration attacks
  and allow distributed/offline ID generation.
- Timezone-aware UTC Datetimes: Explicit timezone tracking for accurate logging and audits.
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime
from app.core.database import Base


def generate_uuid() -> str:
    """Generates a standard UUID4 hex string."""
    return str(uuid.uuid4())


def utc_now() -> datetime:
    """Returns the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class BaseModelMixin:
    """
    Abstract mixin adding standard primary key and timestamp fields.
    """
    id = Column(String(36), primary_key=True, default=generate_uuid, index=True)
    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(
        DateTime(timezone=True),
        default=utc_now,
        onupdate=utc_now,
        nullable=False
    )
