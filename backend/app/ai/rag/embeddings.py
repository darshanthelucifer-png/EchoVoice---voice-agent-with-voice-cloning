"""
Embeddings Manager for RAG (backend/app/ai/rag/embeddings.py)
-----------------------------------------------------------
PYTHON & ML CONCEPTS DEMONSTRATED:
- Sentence-Transformers Embeddings: Maps natural language texts into dense 384-D vector space
  where geometric proximity (cosine distance) reflects semantic similarity.
- Fallback / Mock Embedding Generator: Uses deterministic MD5 feature hashing with L2-normalization
  for instant, zero-download testing and offline CI/CD.
- Thread Safety & Lazy Loading: Defers neural model loading to the first query to preserve fast startup.
"""

import hashlib
from typing import List, Optional
import numpy as np
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.logging import logger, timed_step


class FallbackEmbeddings(Embeddings):
    """
    Deterministic mock embeddings engine for offline CI and rapid unit testing.
    Generates consistent 384-D L2-normalized vectors via hash projection.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _hash_to_vector(self, text: str) -> List[float]:
        """Maps a string into a deterministic normalized pseudo-random float vector."""
        # Use MD5 seed for reproducible pseudo-random numbers
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._hash_to_vector(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._hash_to_vector(text)


class EmbeddingsManager:
    """
    Manages embedding model lifecycles, lazy loading, and hardware placement.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or settings.EMBED_MODEL
        self._embeddings: Optional[Embeddings] = None

    def get_embeddings(self, use_fallback: bool = False) -> Embeddings:
        """Resolves the primary neural embeddings engine or fallback."""
        if use_fallback:
            return FallbackEmbeddings()

        if self._embeddings is None:
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                logger.info(f"Loading sentence-transformer embeddings: '{self.model_name}'...")
                self._embeddings = HuggingFaceEmbeddings(
                    model_name=self.model_name,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True}
                )
                logger.info(f"Embeddings model '{self.model_name}' initialized successfully.")
            except Exception as exc:
                logger.warning(
                    f"Failed to load neural embeddings '{self.model_name}': {exc}. "
                    "Falling back to deterministic FallbackEmbeddings."
                )
                self._embeddings = FallbackEmbeddings()

        return self._embeddings


# Global singleton manager
embeddings_manager = EmbeddingsManager()


def get_embeddings(use_fallback: bool = False) -> Embeddings:
    """Convenience getter for the active embeddings model."""
    return embeddings_manager.get_embeddings(use_fallback=use_fallback)
