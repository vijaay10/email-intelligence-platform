"""
Embedding Provider  (Phase 1 — RAG)
===================================
Wraps a Sentence-Transformers model (default ``all-MiniLM-L6-v2``) to turn
email text into dense vectors for semantic search.

Design
------
* **Lazy + graceful.** The heavy ``sentence-transformers`` import only happens
  on first use. If the package (or the model) is unavailable, :meth:`is_available`
  returns ``False`` and the rest of the platform keeps working — callers check
  availability or catch :class:`EmbeddingError`.
* **Batch-first.** :meth:`embed_texts` encodes a whole list in one call, which is
  far faster than per-text encoding. :meth:`embed_text` is a thin convenience
  wrapper around it.
* **Normalised vectors.** Embeddings are L2-normalised so cosine similarity in
  ChromaDB reduces to an inner product.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence

import config

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised when an embedding is requested but the backend is unavailable."""


class EmbeddingProvider:
    """
    Sentence-Transformers embedding backend.

    Parameters
    ----------
    model_name : str
        Hugging Face / Sentence-Transformers model id. Defaults to
        ``config.EMBEDDING_MODEL`` (``all-MiniLM-L6-v2``).
    batch_size : int
        Batch size for :meth:`embed_texts`. Defaults to
        ``config.EMBEDDING_BATCH_SIZE``.
    """

    def __init__(self, model_name: Optional[str] = None,
                 batch_size: Optional[int] = None):
        self.model_name = model_name or config.EMBEDDING_MODEL
        self.batch_size = batch_size or config.EMBEDDING_BATCH_SIZE
        self._model = None
        self._dimension: Optional[int] = None
        self._unavailable_reason: Optional[str] = None

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True if a model is loaded or can be loaded right now."""
        if self._model is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        return self._try_load()

    @property
    def unavailable_reason(self) -> Optional[str]:
        """Human-readable explanation when :meth:`is_available` is False."""
        return self._unavailable_reason

    @property
    def dimension(self) -> Optional[int]:
        """Embedding dimensionality (loads the model if needed)."""
        self.is_available()
        return self._dimension

    def _try_load(self) -> bool:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:  # ImportError or transitive failure
            self._unavailable_reason = (
                f"sentence-transformers not available: {exc}"
            )
            logger.warning("[Embeddings] %s", self._unavailable_reason)
            return False

        try:
            logger.info("[Embeddings] Loading model '%s'…", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            self._dimension = int(self._model.get_sentence_embedding_dimension())
            logger.info("[Embeddings] Ready (dim=%d).", self._dimension)
            return True
        except Exception as exc:
            self._unavailable_reason = (
                f"could not load model '{self.model_name}': {exc}"
            )
            logger.error("[Embeddings] %s", self._unavailable_reason)
            return False

    # ── Encoding ──────────────────────────────────────────────────────────

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Encode a list of texts into a list of float vectors (batched).

        Raises
        ------
        EmbeddingError
            If the embedding backend is unavailable.
        """
        if not texts:
            return []
        if not self.is_available():
            raise EmbeddingError(
                self._unavailable_reason or "embedding backend unavailable"
            )

        vectors = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return [v.tolist() for v in vectors]

    def embed_text(self, text: str) -> List[float]:
        """Encode a single text into one float vector."""
        return self.embed_texts([text])[0]
