"""
RAG Service  (Phase 1)
======================
Ties the embedding provider, the ChromaDB vector store, and the SQLite
``email_embeddings`` table into one cohesive API:

* :meth:`index_email`      — embed and store a single email (idempotent).
* :meth:`backfill_from_db` — embed every stored email that needs it (batched).
* :meth:`search`           — semantic search over the indexed emails.

Caching / no-redundant-work
---------------------------
Each email's text is hashed (SHA-256). The hash + model name are recorded in
``email_embeddings``. On re-index, if the hash and model are unchanged the email
is skipped — so we never recompute an embedding for content that hasn't changed.

Dependencies are injected (DB, embedder, store), so the service is easy to test
with fakes and easy to reconfigure. All components degrade gracefully: if the
embedder or store is unavailable, :meth:`is_available` is False and the indexing
methods become safe no-ops that report *why*.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Dict, List, Optional

import config
from src.database import DatabaseManager
from src.rag.embeddings import EmbeddingProvider, EmbeddingError
from src.rag.vector_store import VectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class RagService:
    """
    Parameters
    ----------
    db : DatabaseManager, optional
        Database manager. A default instance is created if omitted.
    embedder : EmbeddingProvider, optional
        Embedding backend. A default instance is created if omitted.
    store : VectorStore, optional
        Vector store. A default instance is created if omitted.
    """

    def __init__(
        self,
        db: Optional[DatabaseManager] = None,
        embedder: Optional[EmbeddingProvider] = None,
        store: Optional[VectorStore] = None,
    ):
        self.db = db if db is not None else DatabaseManager()
        self.embedder = embedder if embedder is not None else EmbeddingProvider()
        self.store = store if store is not None else VectorStore()

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True only if BOTH the embedder and the vector store are usable."""
        return self.embedder.is_available() and self.store.is_available()

    def unavailable_reason(self) -> Optional[str]:
        if not self.embedder.is_available():
            return self.embedder.unavailable_reason
        if not self.store.is_available():
            return self.store.unavailable_reason
        return None

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _compose_text(subject: str, body: str) -> str:
        """The text that actually gets embedded (subject + body)."""
        return f"{subject or ''}\n\n{body or ''}".strip()

    @staticmethod
    def _content_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def _chroma_id(email_id: int) -> str:
        return f"email_{email_id}"

    def _needs_embedding(self, email_id: int, content_hash: str, force: bool) -> bool:
        if force:
            return True
        record = self.db.get_embedding_record(email_id)
        if record is None:
            return True
        return (
            record["content_hash"] != content_hash
            or record["embedding_model"] != self.embedder.model_name
        )

    @staticmethod
    def _metadata(email_id: int, sender: str, subject: str, timestamp: str) -> Dict:
        return {
            "email_id": email_id,
            "sender": sender or "",
            "subject": subject or "",
            "timestamp": timestamp or "",
        }

    # ── Single email ──────────────────────────────────────────────────────

    def index_email(
        self,
        email_id: int,
        sender: str,
        subject: str,
        body: str,
        timestamp: str = "",
        force: bool = False,
    ) -> bool:
        """
        Embed and store one email.

        Returns
        -------
        bool
            ``True`` if an embedding was (re)created, ``False`` if it was
            skipped because the content is unchanged or the backend is
            unavailable.
        """
        if not self.is_available():
            logger.warning("[RAG] Skipping index — %s", self.unavailable_reason())
            return False

        text = self._compose_text(subject, body)
        if not text:
            return False

        content_hash = self._content_hash(text)
        if not self._needs_embedding(email_id, content_hash, force):
            logger.info("[RAG] email_id=%d unchanged — skip.", email_id)
            return False

        try:
            vector = self.embedder.embed_text(text)
            chroma_id = self._chroma_id(email_id)
            self.store.upsert(
                ids=[chroma_id],
                embeddings=[vector],
                documents=[text],
                metadatas=[self._metadata(email_id, sender, subject, timestamp)],
            )
            self.db.save_embedding_record(
                email_id=email_id,
                chroma_id=chroma_id,
                content_hash=content_hash,
                embedding_model=self.embedder.model_name,
                vector_dim=len(vector),
            )
            return True
        except (EmbeddingError, VectorStoreError) as exc:
            logger.error("[RAG] index_email failed for %d: %s", email_id, exc)
            return False

    # ── Batch backfill ────────────────────────────────────────────────────

    def backfill_from_db(self, force: bool = False) -> Dict[str, int]:
        """
        Embed every stored email that needs it, using a single batched encode.

        Returns a stats dict: ``{total, embedded, skipped}``.
        """
        emails = self.db.get_all_emails()
        stats = {"total": len(emails), "embedded": 0, "skipped": 0}

        if not self.is_available():
            logger.warning("[RAG] Backfill skipped — %s", self.unavailable_reason())
            stats["skipped"] = len(emails)
            return stats

        # Decide which emails need (re)embedding.
        pending = []  # (email_dict, text, content_hash)
        for email in emails:
            text = self._compose_text(email.get("subject", ""),
                                      email.get("email_body", ""))
            if not text:
                continue
            content_hash = self._content_hash(text)
            if self._needs_embedding(email["id"], content_hash, force):
                pending.append((email, text, content_hash))

        stats["skipped"] = stats["total"] - len(pending)
        if not pending:
            return stats

        try:
            vectors = self.embedder.embed_texts([t for _, t, _ in pending])
        except EmbeddingError as exc:
            logger.error("[RAG] Batch embed failed: %s", exc)
            stats["skipped"] = stats["total"]
            return stats

        ids, embeddings, documents, metadatas = [], [], [], []
        for (email, text, _hash), vector in zip(pending, vectors):
            ids.append(self._chroma_id(email["id"]))
            embeddings.append(vector)
            documents.append(text)
            metadatas.append(self._metadata(
                email["id"], email.get("sender", ""),
                email.get("subject", ""), email.get("date", ""),
            ))

        try:
            self.store.upsert(ids, embeddings, documents, metadatas)
        except VectorStoreError as exc:
            logger.error("[RAG] Batch upsert failed: %s", exc)
            stats["skipped"] = stats["total"]
            return stats

        for (email, _text, content_hash), vector in zip(pending, vectors):
            self.db.save_embedding_record(
                email_id=email["id"],
                chroma_id=self._chroma_id(email["id"]),
                content_hash=content_hash,
                embedding_model=self.embedder.model_name,
                vector_dim=len(vector),
            )
        stats["embedded"] = len(pending)
        logger.info("[RAG] Backfill complete: %s", stats)
        return stats

    # ── Semantic search ───────────────────────────────────────────────────

    def search(self, query: str, top_k: Optional[int] = None) -> List[Dict]:
        """
        Semantic search over indexed emails.

        Returns a list of ``{id, document, metadata, distance, score}`` dicts
        ordered most-relevant first. Returns ``[]`` if the backend is
        unavailable or the query is empty.
        """
        top_k = top_k or config.RAG_TOP_K
        if not query or not query.strip():
            return []
        if not self.is_available():
            logger.warning("[RAG] Search skipped — %s", self.unavailable_reason())
            return []

        try:
            query_vector = self.embedder.embed_text(query)
            hits = self.store.query(query_vector, top_k=top_k)
        except (EmbeddingError, VectorStoreError) as exc:
            logger.error("[RAG] Search failed: %s", exc)
            return []

        # Add a friendly similarity score (1 - cosine distance).
        for hit in hits:
            dist = hit.get("distance")
            hit["score"] = round(1.0 - dist, 4) if isinstance(dist, (int, float)) else None
        return hits

    # ── Diagnostics ───────────────────────────────────────────────────────

    def stats(self) -> Dict:
        return {
            "available": self.is_available(),
            "reason": self.unavailable_reason(),
            "embedding_model": self.embedder.model_name,
            "tracked_in_db": self.db.count_embeddings(),
            "vectors_in_store": self.store.count(),
        }
