"""
RAG Context Retriever & Prompt Formatter (backend/app/ai/rag/retriever.py)
--------------------------------------------------------------------------
PYTHON & RAG CONCEPTS DEMONSTRATED:
- Context Augmentation: Retrieves dense semantic matches from the vector database and formats
  them with explicit citations into an LLM-ready context block.
- Telemetry & Timing: Captures end-to-end vector search latency and similarity scores.
- Dynamic Score Thresholding: Filters low-confidence or irrelevant matches to prevent LLM hallucinations.
"""

from dataclasses import dataclass, field
import time
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.rag.vector_store import vector_store_manager


@dataclass
class RetrievedChunk:
    """Represents a single relevant chunk retrieved from the vector store."""
    text: str
    score: float
    doc_id: str
    title: str
    chunk_index: int
    page: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Standardized retrieval output containing citations, telemetry, and formatted context."""
    query: str
    chunks: List[RetrievedChunk]
    context_text: str
    latency_ms: float
    chunk_count: int
    top_score: float

    @property
    def has_context(self) -> bool:
        return len(self.chunks) > 0


class RAGRetriever:
    """
    Coordinates semantic vector retrieval and context synthesis for conversational agents.
    """

    def __init__(self):
        self.default_k = settings.RAG_TOP_K
        self.default_threshold = settings.RAG_SCORE_THRESHOLD

    @timed_step("RAG Vector Retrieval")
    async def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        user_id: str = "default",
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> RetrievalResult:
        """
        Retrieves top-k relevant chunks from FAISS for the specified user and collection.
        """
        start_t = time.perf_counter()
        top_k = k or self.default_k
        threshold = score_threshold if score_threshold is not None else self.default_threshold

        scored_docs = vector_store_manager.similarity_search_with_score(
            query=query,
            k=top_k,
            score_threshold=threshold,
            user_id=user_id,
            collection_name=collection_name,
            use_fallback=use_fallback
        )

        chunks: List[RetrievedChunk] = []
        for doc, score in scored_docs:
            meta = doc.metadata or {}
            chunks.append(
                RetrievedChunk(
                    text=doc.page_content,
                    score=score,
                    doc_id=str(meta.get("doc_id", "unknown")),
                    title=str(meta.get("title", "Untitled Document")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    page=int(meta.get("page", 1)),
                    metadata=meta
                )
            )

        latency_ms = round((time.perf_counter() - start_t) * 1000.0, 2)
        top_score = chunks[0].score if chunks else 0.0
        context_text = self.format_context(chunks)

        return RetrievalResult(
            query=query,
            chunks=chunks,
            context_text=context_text,
            latency_ms=latency_ms,
            chunk_count=len(chunks),
            top_score=top_score
        )

    @staticmethod
    def format_context(chunks: List[RetrievedChunk]) -> str:
        """
        Formats retrieved chunks into a clean, cited markdown context block for LLM prompts.
        """
        if not chunks:
            return ""

        formatted_blocks: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            source_header = f"[Source {i}: {chunk.title} (Page {chunk.page}, Match: {int(chunk.score * 100)}%)]"
            formatted_blocks.append(f"{source_header}\n{chunk.text}")

        return "\n\n".join(formatted_blocks)


# Global singleton retriever
rag_retriever = RAGRetriever()
