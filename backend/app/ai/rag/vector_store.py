"""
FAISS Vector Store Manager (backend/app/ai/rag/vector_store.py)
--------------------------------------------------------------
PYTHON & VECTOR DB CONCEPTS DEMONSTRATED:
- FAISS (Facebook AI Similarity Search): Blazing-fast CPU-optimized vector index
  for billion-scale dense vector nearest-neighbor search.
- Index Persistence: Serializes FAISS binary graphs and LangChain docstore pickles
  to disk for persistent offline storage.
- Multi-Tenant Isolation: Shards vector stores by `user_id` and `collection_name`
  preventing data leakage across users.
- Metric Transformation: Converts FAISS L2 squared distances into normalized
  cosine similarity scores [0.0, 1.0].
"""

import os
from pathlib import Path
import shutil
import threading
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document as LCDocument
from langchain_community.vectorstores import FAISS

from app.core.config import settings
from app.core.logging import logger, timed_step
from app.ai.rag.embeddings import get_embeddings


class FAISSVectorStoreManager:
    """
    Coordinates creation, ingestion, similarity retrieval, and persistence
    for FAISS vector indices.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = base_dir or settings.VECTOR_STORE_DIR
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._stores: Dict[str, FAISS] = {}
        self._lock = threading.RLock()

    def _get_store_key(self, user_id: str, collection_name: str) -> str:
        return f"{user_id}::{collection_name}"

    def _get_index_dir(self, user_id: str, collection_name: str) -> Path:
        return self.base_dir / user_id / collection_name

    def load_or_get_store(
        self,
        user_id: str,
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> Optional[FAISS]:
        """Loads an existing FAISS store from disk or returns cached in-memory instance."""
        key = self._get_store_key(user_id, collection_name)
        with self._lock:
            if key in self._stores:
                return self._stores[key]

            index_dir = self._get_index_dir(user_id, collection_name)
            faiss_file = index_dir / "index.faiss"
            pkl_file = index_dir / "index.pkl"

            if faiss_file.exists() and pkl_file.exists():
                try:
                    embeddings = get_embeddings(use_fallback=use_fallback)
                    store = FAISS.load_local(
                        folder_path=str(index_dir),
                        embeddings=embeddings,
                        allow_dangerous_deserialization=True
                    )
                    self._stores[key] = store
                    logger.info(f"Loaded existing FAISS vector store for {key} from '{index_dir}'")
                    return store
                except Exception as exc:
                    logger.error(f"Failed to load FAISS index from {index_dir}: {exc}")
                    return None

            return None

    def add_documents(
        self,
        documents: List[LCDocument],
        user_id: str,
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> int:
        """
        Adds document chunks to the FAISS store and saves the updated index to disk.
        Returns the number of added chunks.
        """
        if not documents:
            return 0

        key = self._get_store_key(user_id, collection_name)
        index_dir = self._get_index_dir(user_id, collection_name)
        index_dir.mkdir(parents=True, exist_ok=True)

        embeddings = get_embeddings(use_fallback=use_fallback)

        with self._lock:
            store = self._stores.get(key)
            if store is None:
                faiss_file = index_dir / "index.faiss"
                if faiss_file.exists():
                    try:
                        store = FAISS.load_local(
                            folder_path=str(index_dir),
                            embeddings=embeddings,
                            allow_dangerous_deserialization=True
                        )
                    except Exception:
                        store = None

            if store is None:
                logger.info(f"Creating new FAISS vector store for {key} with {len(documents)} chunks...")
                store = FAISS.from_documents(documents, embeddings)
            else:
                logger.info(f"Adding {len(documents)} chunks to existing FAISS store for {key}...")
                store.add_documents(documents)

            # Persist index to disk
            store.save_local(str(index_dir))
            self._stores[key] = store

        return len(documents)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        score_threshold: Optional[float] = None,
        user_id: str = "default",
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> List[Tuple[LCDocument, float]]:
        """
        Executes dense vector similarity search.
        Returns list of (Document, cosine_similarity_score) where score is in [0.0, 1.0].
        """
        store = self.load_or_get_store(
            user_id=user_id,
            collection_name=collection_name,
            use_fallback=use_fallback
        )
        if store is None:
            return []

        # FAISS search with L2 distance
        # For normalized embeddings: L2_dist_sq = 2 * (1 - cosine_similarity)
        # Therefore: cosine_similarity = 1.0 - (L2_dist_sq / 2.0)
        results = store.similarity_search_with_score(query, k=k)

        scored_results: List[Tuple[LCDocument, float]] = []
        for doc, raw_distance in results:
            # Map distance to similarity [0, 1]
            cosine_sim = max(0.0, min(1.0, 1.0 - (float(raw_distance) / 2.0)))
            if score_threshold is not None and cosine_sim < score_threshold:
                continue
            scored_results.append((doc, round(cosine_sim, 4)))

        # Sort descending by similarity score
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return scored_results

    def delete_document_chunks(
        self,
        document_id: str,
        user_id: str,
        collection_name: str = "default",
        use_fallback: bool = False
    ) -> int:
        """
        Removes all vector chunks belonging to a specific document ID.
        Rebuilds the collection index if remaining chunks exist.
        """
        store = self.load_or_get_store(
            user_id=user_id,
            collection_name=collection_name,
            use_fallback=use_fallback
        )
        if store is None:
            return 0

        key = self._get_store_key(user_id, collection_name)
        index_dir = self._get_index_dir(user_id, collection_name)

        with self._lock:
            # Extract all documents in the docstore
            remaining_docs: List[LCDocument] = []
            deleted_count = 0

            # Access internal docstore
            docstore = store.docstore
            if hasattr(docstore, "_dict"):
                for doc in docstore._dict.values():
                    if isinstance(doc, LCDocument):
                        if doc.metadata.get("doc_id") == document_id:
                            deleted_count += 1
                        else:
                            remaining_docs.append(doc)

            # Re-index or clear
            if not remaining_docs:
                self.clear_collection(user_id=user_id, collection_name=collection_name)
            else:
                embeddings = get_embeddings(use_fallback=use_fallback)
                new_store = FAISS.from_documents(remaining_docs, embeddings)
                new_store.save_local(str(index_dir))
                self._stores[key] = new_store

        return deleted_count

    def clear_collection(self, user_id: str, collection_name: str = "default") -> None:
        """Wipes the FAISS index files for a collection from disk and memory."""
        key = self._get_store_key(user_id, collection_name)
        index_dir = self._get_index_dir(user_id, collection_name)

        with self._lock:
            if key in self._stores:
                del self._stores[key]

            if index_dir.exists():
                shutil.rmtree(index_dir, ignore_errors=True)

    def get_collection_stats(self, user_id: str, collection_name: str = "default") -> Dict[str, Any]:
        """Returns statistics for a vector store collection."""
        index_dir = self._get_index_dir(user_id, collection_name)
        store = self.load_or_get_store(user_id=user_id, collection_name=collection_name)

        chunk_count = 0
        if store and hasattr(store, "docstore") and hasattr(store.docstore, "_dict"):
            chunk_count = len(store.docstore._dict)

        size_bytes = 0
        if index_dir.exists():
            for f in index_dir.glob("*"):
                if f.is_file():
                    size_bytes += f.stat().st_size

        return {
            "collection_name": collection_name,
            "chunk_count": chunk_count,
            "size_bytes": size_bytes,
            "exists": index_dir.exists() and (index_dir / "index.faiss").exists()
        }


# Global singleton manager
vector_store_manager = FAISSVectorStoreManager()
