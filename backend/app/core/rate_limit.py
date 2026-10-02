"""
Rate Limiting Module (backend/app/core/rate_limit.py)
-----------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Singleton Pattern: Configures a single shared `Limiter` instance used across
  all API route decorators.
- Middleware integration with `slowapi` to protect free Hugging Face API
  endpoints and database routes from denial-of-service / brute-force traffic.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

# Initialize limiter configured with client IP key
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[settings.RATE_LIMIT_DEFAULT],
    storage_uri="memory://",  # Upgradeable to redis:// via config when scaled
)
