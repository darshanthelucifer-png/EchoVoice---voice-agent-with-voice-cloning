"""
User Schemas (backend/app/schemas/user.py)
------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic v2 `BaseModel` & `ConfigDict(from_attributes=True)`: Enables direct
  serialization from SQLAlchemy ORM models to JSON API output.
- Field Validation & EmailStr: Built-in validation for emails and password lengths.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters long")
    full_name: Optional[str] = None


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    password: Optional[str] = Field(None, min_length=8)
    is_active: Optional[bool] = None


class UserRead(UserBase):
    id: str
    is_superuser: bool
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
