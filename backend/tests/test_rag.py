"""
RAG System Unit & Integration Tests (backend/tests/test_rag.py)
---------------------------------------------------------------
PYTHON & RAG CONCEPTS TESTED:
1. Fallback & Neural Embeddings: Verifies 384-D vector generation and L2 normalization.
2. Document Parsing: Extracts structural text sections from raw byte streams.
3. Recursive Text Splitter: Tests hierarchical character chunking and metadata preservation.
4. FAISS Vector Store: Tests persistent index creation, vector similarity search, and chunk deletion.
5. Context Synthesis & Citations: Tests assembled prompt context blocks and similarity score thresholds.
6. Service & REST API Lifecycle: Ingestion, listing, retrieval query, statistics, and deletion.
"""

import io
import shutil
from pathlib import Path
import pytest
import numpy as np
from httpx import AsyncClient

from app.core.config import settings
from app.ai.rag.embeddings import FallbackEmbeddings, get_embeddings
from app.ai.rag.parser import document_parser
from app.ai.rag.splitter import rag_text_splitter
from app.ai.rag.vector_store import FAISSVectorStoreManager
from app.ai.rag.retriever import RAGRetriever, RetrievedChunk
from app.services.rag_service import rag_service


# ==============================================================================
# 1. Embeddings & Parser Tests
# ==============================================================================

def test_fallback_embeddings():
    """Verify deterministic mock embeddings generate 384-D L2-normalized vectors."""
    emb = FallbackEmbeddings(dim=384)
    vec1 = emb.embed_query("EchoVoice conversational voice agent")
    vec2 = emb.embed_query("EchoVoice conversational voice agent")
    vec3 = emb.embed_query("Completely unrelated query")

    assert len(vec1) == 384
    # Deterministic output for same input
    assert vec1 == vec2
    # Distinct output for different input
    assert vec1 != vec3

    # L2-norm must be 1.0 (unit vector)
    norm = np.linalg.norm(np.array(vec1, dtype=np.float32))
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_document_parser_text_and_markdown():
    """Verify parser extracts text and preserves page metadata from raw buffers."""
    content = (
        "# User Manual\n\n"
        "EchoVoice is an open-source real-time voice assistant with RAG.\n"
        "It supports conversational interactions with sub-second latency."
    ).encode("utf-8")

    sections = document_parser.parse_file(content, filename="manual.md")
    assert len(sections) == 1
    assert "EchoVoice" in sections[0].text
    assert sections[0].page_number == 1


# ==============================================================================
# 2. Text Splitter Tests
# ==============================================================================

def test_rag_text_splitter_multilingual():
    """Verify recursive splitting with Western and Indian punctuation delimiters."""
    multilingual_text = (
        "EchoVoice provides real-time voice cloning. "
        "ध्वनि क्लोनिंग प्रणाली अत्यंत तीव्र आणि अचूक आहे। "
        "Every utterance is verified against a strict quality rubric. "
        "प्रत्येक ऑडिओ क्लिपचे विश्लेषण केले जाते॥"
    )

    chunks = rag_text_splitter.split_raw_text(
        text=multilingual_text,
        document_id="doc_test_split",
        document_title="Multilingual Voice Docs",
        collection_name="test_col"
    )

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) > 0
        assert chunk.metadata["doc_id"] == "doc_test_split"
        assert chunk.metadata["title"] == "Multilingual Voice Docs"
        assert chunk.metadata["collection"] == "test_col"


# ==============================================================================
# 3. FAISS Vector Store Tests
# ==============================================================================

def test_faiss_vector_store_crud(tmp_path: Path):
    """
    Verify FAISS vector store creation, persistence, query retrieval,
    and document chunk deletion using isolated test directory.
    """
    vsm = FAISSVectorStoreManager(base_dir=tmp_path)
    user_id = "user_test_faiss"
    collection = "kb_test"

    text = (
        "Retrieval-Augmented Generation enhances LLM responses with factual knowledge. "
        "FAISS stores dense embeddings and calculates cosine similarity in milliseconds."
    )
    chunks = rag_text_splitter.split_raw_text(
        text=text,
        document_id="doc_faiss_1",
        document_title="RAG Primer",
        collection_name=collection
    )

    # 1. Ingest chunks (use FallbackEmbeddings for speed)
    added = vsm.add_documents(chunks, user_id=user_id, collection_name=collection, use_fallback=True)
    assert added == len(chunks)

    # 2. Verify disk persistence
    index_file = tmp_path / user_id / collection / "index.faiss"
    pkl_file = tmp_path / user_id / collection / "index.pkl"
    assert index_file.exists()
    assert pkl_file.exists()

    # 3. Verify similarity search
    results = vsm.similarity_search_with_score(
        query="Retrieval-Augmented Generation",
        k=2,
        user_id=user_id,
        collection_name=collection,
        use_fallback=True
    )
    assert len(results) > 0
    top_doc, top_score = results[0]
    assert "Retrieval-Augmented" in top_doc.page_content
    assert 0.0 <= top_score <= 1.0

    # 4. Verify stats
    stats = vsm.get_collection_stats(user_id=user_id, collection_name=collection)
    assert stats["chunk_count"] > 0
    assert stats["size_bytes"] > 0
    assert stats["exists"] is True

    # 5. Delete document chunks
    deleted = vsm.delete_document_chunks(
        document_id="doc_faiss_1",
        user_id=user_id,
        collection_name=collection,
        use_fallback=True
    )
    assert deleted > 0

    # Query after deletion should return no results
    post_del_results = vsm.similarity_search_with_score(
        query="Retrieval-Augmented Generation",
        user_id=user_id,
        collection_name=collection,
        use_fallback=True
    )
    assert len(post_del_results) == 0


# ==============================================================================
# 4. Context Formatting & Telemetry Tests
# ==============================================================================

def test_rag_retriever_context_formatting():
    """Verify prompt context assembly contains numbered source citations and page numbers."""
    retriever = RAGRetriever()
    chunks = [
        RetrievedChunk(
            text="The microphone input is processed through Silero VAD.",
            score=0.92,
            doc_id="doc_1",
            title="Audio Pipeline Guide",
            chunk_index=0,
            page=2
        ),
        RetrievedChunk(
            text="XTTS-v2 produces synthetic speech conditioned on a speaker embedding.",
            score=0.88,
            doc_id="doc_2",
            title="TTS Architecture",
            chunk_index=1,
            page=4
        ),
    ]

    context = retriever.format_context(chunks)
    assert "[Source 1: Audio Pipeline Guide (Page 2, Match: 92%)]" in context
    assert "Silero VAD" in context
    assert "[Source 2: TTS Architecture (Page 4, Match: 88%)]" in context
    assert "XTTS-v2" in context


# ==============================================================================
# 5. REST API Endpoint Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_rag_api_lifecycle(client: AsyncClient):
    """
    Test full REST API lifecycle:
    1. Register user and authenticate
    2. Ingest raw text document via POST /api/v1/rag/documents/raw
    3. List documents via GET /api/v1/rag/documents
    4. Execute semantic query via POST /api/v1/rag/query
    5. Query collection statistics via GET /api/v1/rag/stats
    6. Delete document via DELETE /api/v1/rag/documents/{id}
    """
    # 1. Register test user
    user_payload = {
        "email": "rag_tester@echovoice.ai",
        "password": "Password123!",
        "full_name": "RAG Tester"
    }
    reg_res = await client.post("/api/v1/auth/register", json=user_payload)
    assert reg_res.status_code in (200, 201)
    token = reg_res.json()["data"]["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Ingest raw text document
    ingest_payload = {
        "title": "EchoVoice Safety Guidelines",
        "content": (
            "Safety Guideline 1: Explicit verbal or written consent is mandatory before voice enrollment. "
            "Safety Guideline 2: Audio models run on open-source weights with zero vendor lock-in. "
            "Safety Guideline 3: All intermediate audio files are mastered to -14 LUFS."
        ),
        "collection_name": "safety"
    }
    ingest_res = await client.post("/api/v1/rag/documents/raw", headers=headers, json=ingest_payload)
    assert ingest_res.status_code == 200
    doc_data = ingest_res.json()["data"]
    doc_id = doc_data["id"]
    assert doc_data["title"] == "EchoVoice Safety Guidelines"
    assert doc_data["chunk_count"] > 0
    assert doc_data["is_indexed"] is True

    # 3. List documents
    list_res = await client.get("/api/v1/rag/documents?collection_name=safety", headers=headers)
    assert list_res.status_code == 200
    docs = list_res.json()["data"]
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id

    # 4. Query knowledge base
    query_payload = {
        "query": "What are the rules regarding consent and voice enrollment?",
        "top_k": 2,
        "collection_name": "safety"
    }
    query_res = await client.post("/api/v1/rag/query", headers=headers, json=query_payload)
    assert query_res.status_code == 200
    q_data = query_res.json()["data"]
    assert q_data["chunk_count"] > 0
    assert "consent" in q_data["context_text"].lower()
    assert len(q_data["chunks"]) > 0

    # 5. Check statistics
    stats_res = await client.get("/api/v1/rag/stats?collection_name=safety", headers=headers)
    assert stats_res.status_code == 200
    s_data = stats_res.json()["data"]
    assert s_data["document_count"] == 1
    assert s_data["chunk_count"] > 0

    # 6. Delete document
    del_res = await client.delete(f"/api/v1/rag/documents/{doc_id}", headers=headers)
    assert del_res.status_code == 200
    assert del_res.json()["data"] is True

    # 7. Verify document is gone
    get_res = await client.get(f"/api/v1/rag/documents/{doc_id}", headers=headers)
    assert get_res.status_code == 404
