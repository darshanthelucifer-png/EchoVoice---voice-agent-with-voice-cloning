"""
EchoVoice RAG & Vector Knowledge Package (backend/app/ai/rag/__init__.py)
-----------------------------------------------------------------------
Exports core document parsing, chunking, FAISS vector indexing, and semantic retrieval interfaces.
"""

from app.ai.rag.embeddings import get_embeddings, FallbackEmbeddings, embeddings_manager
from app.ai.rag.parser import document_parser, DocumentParser, ParsedSection
from app.ai.rag.splitter import rag_text_splitter, RAGTextSplitter
from app.ai.rag.vector_store import vector_store_manager, FAISSVectorStoreManager
from app.ai.rag.retriever import rag_retriever, RAGRetriever, RetrievedChunk, RetrievalResult

__all__ = [
    "get_embeddings",
    "FallbackEmbeddings",
    "embeddings_manager",
    "document_parser",
    "DocumentParser",
    "ParsedSection",
    "rag_text_splitter",
    "RAGTextSplitter",
    "vector_store_manager",
    "FAISSVectorStoreManager",
    "rag_retriever",
    "RAGRetriever",
    "RetrievedChunk",
    "RetrievalResult",
]
