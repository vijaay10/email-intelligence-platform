"""
Vector Store  (Phase 1 — RAG)
=============================
Thin wrapper around a persistent ChromaDB collection that holds email
embeddings plus their metadata (sender, subject, timestamp, email_id) and the
original text as the retrievable document.

Like :mod:`src.rag.embeddings`, the ``chromadb`` import is lazy and the wrapper
degrades gracefully: if Chroma is not installed, :meth:`is_available` returns
``False`` and the rest of the platform is unaffected.

We pass our own (already-normalised) embeddings to Chroma and configure the
collection for cosine distance, so we never rely on Chroma's built-in embedder.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import config

logger = logging.getLogger(__name__)


class VectorStoreError(RuntimeError):
    """Raised when a vector-store operation is attempted while unavailable."""


def _sanitise_metadata(meta: Dict) -> Dict:
    """Chroma rejects None metadata values — coerce them to empty strings."""
    clean: Dict = {}
    for key, value in meta.items():
        if value is None:
            clean[key] = ""
        elif isinstance(value, (str, int, float, bool)):
            clean[key] = value
        else:
            clean[key] = str(value)
    return clean


class VectorStore:
    """
    Persistent ChromaDB collection for email embeddings.

    Parameters
    ----------
    persist_dir : str | Path
        On-disk location for the Chroma database. Defaults to
        ``config.CHROMA_DIR``.
    collection_name : str
        Collection name. Defaults to ``config.CHROMA_COLLECTION``.
    distance : str
        Distance metric for the HNSW index (``cosine`` by default).
    """

    def __init__(self, persist_dir=None, collection_name: Optional[str] = None,
                 distance: str = "cosine"):
        self.persist_dir = str(persist_dir or config.CHROMA_DIR)
        self.collection_name = collection_name or config.CHROMA_COLLECTION
        self.distance = distance
        self._client = None
        self._collection = None
        self._unavailable_reason: Optional[str] = None

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        if self._collection is not None:
            return True
        if self._unavailable_reason is not None:
            return False
        return self._try_init()

    @property
    def unavailable_reason(self) -> Optional[str]:
        return self._unavailable_reason

    def _try_init(self) -> bool:
        try:
            import chromadb
        except Exception as exc:
            self._unavailable_reason = f"chromadb not available: {exc}"
            logger.warning("[VectorStore] %s", self._unavailable_reason)
            return False

        try:
            Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.distance},
            )
            logger.info(
                "[VectorStore] Collection '%s' ready at %s (count=%d).",
                self.collection_name, self.persist_dir, self._collection.count(),
            )
            return True
        except Exception as exc:
            self._unavailable_reason = f"could not open Chroma store: {exc}"
            logger.error("[VectorStore] %s", self._unavailable_reason)
            return False

    def _require(self) -> None:
        if not self.is_available():
            raise VectorStoreError(
                self._unavailable_reason or "vector store unavailable"
            )

    # ── Write ─────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Dict],
    ) -> None:
        """Insert or update a batch of vectors. No-op for empty input."""
        if not ids:
            return
        self._require()
        self._collection.upsert(
            ids=list(ids),
            embeddings=[list(e) for e in embeddings],
            documents=list(documents),
            metadatas=[_sanitise_metadata(m) for m in metadatas],
        )
        logger.info("[VectorStore] Upserted %d vector(s).", len(ids))

    # ── Read ──────────────────────────────────────────────────────────────

    def query(self, query_embedding: Sequence[float], top_k: int = 5) -> List[Dict]:
        """
        Return up to *top_k* nearest neighbours.

        Each result is a dict: ``{id, document, metadata, distance}``,
        ordered nearest-first.
        """
        self._require()
        res = self._collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
        )
        # Chroma returns each field as a list-of-lists (one per query).
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        results: List[Dict] = []
        for i, _id in enumerate(ids):
            results.append({
                "id": _id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": dists[i] if i < len(dists) else None,
            })
        return results

    def count(self) -> int:
        if not self.is_available():
            return 0
        return self._collection.count()

    # ── Maintenance ───────────────────────────────────────────────────────

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._require()
        self._collection.delete(ids=list(ids))

    def reset(self) -> None:
        """Drop and recreate the collection (used in tests / re-index)."""
        self._require()
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance},
        )
        logger.warning("[VectorStore] Collection '%s' reset.", self.collection_name)
