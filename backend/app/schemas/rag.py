"""
RAG Pydantic Schemas (backend/app/schemas/rag.py)
------------------------------------------------
PYTHON CONCEPTS DEMONSTRATED:
- Pydantic v2 Models: Request validation and serialization for document ingestion,
  vector search queries, and retrieval telemetry.
- Field Constraints: Bounds on top_k, score_threshold, and title lengths.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class DocumentResponse(BaseModel):
    """Metadata response for an ingested knowledge base document."""
    id: str
    user_id: str
    title: str
    file_type: str
    file_size_bytes: int
    chunk_count: int
    collection_name: str
    is_indexed: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class RawDocumentIngestRequest(BaseModel):
    """Request payload for directly ingesting raw text or markdown content."""
    title: str = Field(..., min_length=1, max_length=255, description="Document title")
    content: str = Field(..., min_length=10, description="Raw text or markdown content")
    collection_name: str = Field(default="default", max_length=100, description="Collection partition name")


class RAGQueryRequest(BaseModel):
    """Request payload for semantic similarity query against vector knowledge base."""
    query: str = Field(..., min_length=1, max_length=1000, description="User search query or question")
    top_k: int = Field(default=4, ge=1, le=20, description="Maximum number of chunks to retrieve")
    score_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Minimum cosine similarity")
    collection_name: str = Field(default="default", description="Collection partition to query")


class RetrievedChunkSchema(BaseModel):
    """Schema for individual retrieved context chunk."""
    text: str
    score: float
    doc_id: str
    title: str
    chunk_index: int
    page: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RAGQueryResponse(BaseModel):
    """Comprehensive semantic retrieval response with formatted context and telemetry."""
    query: str
    chunks: List[RetrievedChunkSchema]
    context_text: str
    chunk_count: int
    top_score: float
    latency_ms: float


class RAGStatsResponse(BaseModel):
    """Telemetry and storage statistics for a vector knowledge collection."""
    collection_name: str
    chunk_count: int
    size_bytes: int
    document_count: int
    exists: bool
