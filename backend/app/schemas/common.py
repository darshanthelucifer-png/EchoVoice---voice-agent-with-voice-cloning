"""
Common Schemas (backend/app/schemas/common.py)
----------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic v2 Generic Models (`Generic[T]`): Enables strictly typed API responses
  wrapping any data payload with standardized success, message, and error fields.
"""

from typing import Generic, TypeVar, Optional, Any, Dict
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class APIResponse(BaseModel, Generic[DataT]):
    """Standardized API response envelope."""
    success: bool = True
    message: str = "Operation completed successfully"
    data: Optional[DataT] = None
    errors: Optional[Any] = None


class HealthResponse(BaseModel):
    """Health check payload."""
    status: str = "healthy"
    app_name: str
    version: str
    environment: str
    database_connected: bool
    configured_models: Dict[str, str]
