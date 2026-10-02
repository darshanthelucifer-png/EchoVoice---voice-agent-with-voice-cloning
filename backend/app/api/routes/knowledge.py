"""
Knowledge Base Router (backend/app/api/routes/knowledge.py)
-----------------------------------------------------------
Aliases to the full RAG knowledge router for backward-compatible routing.
"""

from app.api.routes.rag import router

# Export the production RAG router under the knowledge alias
__all__ = ["router"]

