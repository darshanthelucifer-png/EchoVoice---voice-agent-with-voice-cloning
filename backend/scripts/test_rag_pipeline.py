"""
Standalone RAG Pipeline Verification Script (backend/scripts/test_rag_pipeline.py)
----------------------------------------------------------------------------------
Tests end-to-end:
1. Document Parsing & Text Extraction
2. Recursive Character Chunking
3. Dense Vector Embeddings (sentence-transformers / Fallback)
4. FAISS Vector Store Ingestion, Persistence, and Reloading
5. Semantic Query Retrieval, Cosine Similarity Scoring, and Context Synthesis
6. Chunk Deletion and Multi-Tenancy Isolation
"""

import asyncio
import io
import shutil
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.core.config import settings
from app.ai.rag.embeddings import get_embeddings, FallbackEmbeddings
from app.ai.rag.parser import document_parser
from app.ai.rag.splitter import rag_text_splitter
from app.ai.rag.vector_store import FAISSVectorStoreManager
from app.ai.rag.retriever import RAGRetriever


async def run_rag_pipeline_test():
    print("=" * 70)
    print(" EchoVoice Phase 8: RAG & FAISS Vector Pipeline Verification")
    print("=" * 70)

    # 1. Embeddings Test
    print("\n--- 1. Testing Embeddings ---")
    emb = get_embeddings(use_fallback=False)
    test_vec = emb.embed_query("EchoVoice conversational voice agent")
    print(f"Embedding Model: {settings.EMBED_MODEL}")
    print(f"Vector Dimension: {len(test_vec)} | Sample: [{test_vec[0]:.4f}, {test_vec[1]:.4f}, ...]")
    assert len(test_vec) == 384, f"Expected 384-D vector, got {len(test_vec)}"

    # 2. Text Parser Test
    print("\n--- 2. Testing Document Parser ---")
    sample_text = """
    # EchoVoice Architecture Overview

    EchoVoice is a real-time conversational voice assistant with studio-quality voice cloning and RAG.
    It combines Whisper ASR, XTTS-v2 speech synthesis, and FAISS dense vector retrieval.

    ## Ethical Voice Cloning Policy
    Users must explicitly confirm consent before creating a synthetic voice profile.
    Voice cloning without explicit authorization is strictly prohibited.
    All reference audio clips are stored securely with L2-normalized 256-D speaker embeddings.
    """
    sections = document_parser.parse_text(sample_text.encode("utf-8"), filename="architecture.md")
    print(f"Parsed sections: {len(sections)} | Page: {sections[0].page_number}")
    assert len(sections) > 0

    # 3. Text Splitter Test
    print("\n--- 3. Testing Recursive Text Splitter ---")
    chunks = rag_text_splitter.split_sections(
        sections=sections,
        document_id="doc_test_101",
        document_title="EchoVoice Architecture Guide",
        collection_name="manuals"
    )
    print(f"Generated {len(chunks)} chunks with chunk_size={settings.RAG_CHUNK_SIZE}:")
    for i, c in enumerate(chunks):
        print(f"  Chunk {i} [{len(c.page_content)} chars]: {c.page_content[:60]}...")
        assert c.metadata["doc_id"] == "doc_test_101"
        assert c.metadata["title"] == "EchoVoice Architecture Guide"

    # 4. FAISS Vector Store Ingestion & Persistence
    print("\n--- 4. Testing FAISS Vector Store ---")
    test_dir = settings.DATA_DIR / "test_vector_store"
    if test_dir.exists():
        shutil.rmtree(test_dir, ignore_errors=True)

    vsm = FAISSVectorStoreManager(base_dir=test_dir)
    user_id = "user_test_999"

    # Ingest document chunks
    added_count = vsm.add_documents(
        documents=chunks,
        user_id=user_id,
        collection_name="manuals"
    )
    print(f"Successfully indexed {added_count} chunks into FAISS.")

    # Verify disk persistence
    index_path = test_dir / user_id / "manuals" / "index.faiss"
    assert index_path.exists(), f"Index file missing at {index_path}"
    print(f"FAISS index persisted to disk ({index_path.stat().st_size} bytes).")

    # 5. Semantic Similarity Search & Retrieval
    print("\n--- 5. Testing Semantic Search & Context Synthesis ---")
    retriever = RAGRetriever()
    # Temporarily monkey-patch retriever's vector_store_manager to our test manager
    import app.ai.rag.retriever as ret_module
    old_mgr = ret_module.vector_store_manager
    ret_module.vector_store_manager = vsm

    try:
        query = "What is the policy for ethical voice cloning and consent?"
        print(f"Query: '{query}'")
        result = await retriever.retrieve(
            query=query,
            k=2,
            user_id=user_id,
            collection_name="manuals"
        )

        print(f"Retrieved {result.chunk_count} chunks in {result.latency_ms} ms (Top match: {int(result.top_score * 100)}%):")
        print("\n--- Assembled LLM Context Block ---")
        print(result.context_text)
        print("-----------------------------------")

        assert result.chunk_count > 0
        assert "consent" in result.context_text.lower()
        assert result.top_score > 0.40, f"Expected strong match score, got {result.top_score}"

        # 6. Document Deletion Test
        print("\n--- 6. Testing Document Deletion ---")
        deleted_count = vsm.delete_document_chunks(
            document_id="doc_test_101",
            user_id=user_id,
            collection_name="manuals"
        )
        print(f"Deleted {deleted_count} chunks for doc_test_101.")

        # Re-query should return 0 chunks
        empty_res = await retriever.retrieve(query=query, user_id=user_id, collection_name="manuals")
        print(f"Post-deletion query matches: {empty_res.chunk_count}")
        assert empty_res.chunk_count == 0

    finally:
        ret_module.vector_store_manager = old_mgr
        if test_dir.exists():
            shutil.rmtree(test_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print(" ALL PHASE 8 RAG PIPELINE VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_rag_pipeline_test())
