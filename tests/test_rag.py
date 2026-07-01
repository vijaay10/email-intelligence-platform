"""
Unit Tests — RAG layer  (Phase 1)
=================================
Covers the RagService dedup/batch/search logic with lightweight fakes (so the
tests run without sentence-transformers or chromadb installed), plus the
graceful-degradation path of the real EmbeddingProvider / VectorStore.

Run:
    python -m unittest tests.test_rag -v
"""

import os
import sys
import math
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.database import DatabaseManager
from src.services.rag_service import RagService


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeEmbedder:
    """Deterministic, dependency-free stand-in for EmbeddingProvider."""

    model_name = "fake-model"

    def __init__(self):
        self.calls = 0

    def is_available(self) -> bool:
        return True

    @property
    def unavailable_reason(self):
        return None

    def _vec(self, text: str):
        # 4-dim vector from character buckets, L2-normalised — deterministic.
        v = [0.0, 0.0, 0.0, 0.0]
        for ch in text.lower():
            if ch.isalpha():
                v[ord(ch) % 4] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_texts(self, texts):
        self.calls += 1
        return [self._vec(t) for t in texts]

    def embed_text(self, text):
        return self.embed_texts([text])[0]


class FakeStore:
    """In-memory cosine vector store mimicking the VectorStore API."""

    def __init__(self):
        self._data = {}  # id -> (embedding, document, metadata)

    def is_available(self) -> bool:
        return True

    @property
    def unavailable_reason(self):
        return None

    def upsert(self, ids, embeddings, documents, metadatas):
        for _id, emb, doc, meta in zip(ids, embeddings, documents, metadatas):
            self._data[_id] = (list(emb), doc, dict(meta))

    @staticmethod
    def _cosine(a, b):
        return sum(x * y for x, y in zip(a, b))

    def query(self, query_embedding, top_k=5):
        scored = [
            {"id": _id, "document": doc, "metadata": meta,
             "distance": 1.0 - self._cosine(query_embedding, emb)}
            for _id, (emb, doc, meta) in self._data.items()
        ]
        scored.sort(key=lambda h: h["distance"])
        return scored[:top_k]

    def count(self):
        return len(self._data)

    def delete(self, ids):
        for _id in ids:
            self._data.pop(_id, None)


# ── RagService logic ──────────────────────────────────────────────────────────

class TestRagService(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = DatabaseManager(db_path=self.tmp.name)
        self.rag = RagService(db=self.db, embedder=FakeEmbedder(), store=FakeStore())

    def tearDown(self):
        self.db = None
        os.unlink(self.tmp.name)

    def _insert_email(self, subject, body, sender="a@b.com", sentiment="neutral"):
        return self.db.save_email_analysis(
            sender=sender, subject=subject, email_body=body,
            overall_sentiment=sentiment, importance_level="LOW",
            sentiment_shift=False, urgency_score=0.0, date="2025-10-01",
            sentences=[],
        )

    def test_available(self):
        self.assertTrue(self.rag.is_available())

    def test_index_email_creates_then_dedupes(self):
        eid = self._insert_email("Invoice overdue", "Please pay invoice 4521.")
        # First time → embedding created.
        self.assertTrue(self.rag.index_email(eid, "a@b.com", "Invoice overdue",
                                             "Please pay invoice 4521."))
        # Second time, unchanged → skipped.
        self.assertFalse(self.rag.index_email(eid, "a@b.com", "Invoice overdue",
                                              "Please pay invoice 4521."))
        self.assertEqual(self.db.count_embeddings(), 1)

    def test_index_email_reembeds_on_change(self):
        eid = self._insert_email("Hello", "Body one.")
        self.assertTrue(self.rag.index_email(eid, "a@b.com", "Hello", "Body one."))
        # Changed content → re-embedded.
        self.assertTrue(self.rag.index_email(eid, "a@b.com", "Hello", "Body two!"))
        self.assertEqual(self.db.count_embeddings(), 1)  # still one row (upsert)

    def test_force_reembeds(self):
        eid = self._insert_email("Hello", "Body one.")
        self.assertTrue(self.rag.index_email(eid, "a@b.com", "Hello", "Body one."))
        self.assertTrue(self.rag.index_email(eid, "a@b.com", "Hello", "Body one.",
                                             force=True))

    def test_backfill_batches_and_skips(self):
        self._insert_email("One", "First email about kubernetes.")
        self._insert_email("Two", "Second email about invoices.")
        self._insert_email("Three", "Third email about meetings.")

        embedder = self.rag.embedder
        stats = self.rag.backfill_from_db()
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["embedded"], 3)
        self.assertEqual(stats["skipped"], 0)
        # All three embedded in a SINGLE batched encode call.
        self.assertEqual(embedder.calls, 1)
        self.assertEqual(self.db.count_embeddings(), 3)

        # Re-run → nothing to do.
        stats2 = self.rag.backfill_from_db()
        self.assertEqual(stats2["embedded"], 0)
        self.assertEqual(stats2["skipped"], 3)

    def test_search_returns_relevant_results(self):
        self._insert_email("Kubernetes outage", "The kubernetes cluster is down.")
        self._insert_email("Lunch", "Want to grab lunch tomorrow?")
        self.rag.backfill_from_db()

        results = self.rag.search("kubernetes cluster problem", top_k=2)
        self.assertTrue(len(results) >= 1)
        self.assertLessEqual(len(results), 2)
        for hit in results:
            self.assertIn("metadata", hit)
            self.assertIn("score", hit)

    def test_search_empty_query(self):
        self.assertEqual(self.rag.search("   "), [])

    def test_stats_shape(self):
        s = self.rag.stats()
        for key in ("available", "embedding_model", "tracked_in_db", "vectors_in_store"):
            self.assertIn(key, s)


# ── Graceful degradation (real backends, deps absent in this env) ──────────────

class TestGracefulDegradation(unittest.TestCase):
    """
    With the AI packages absent, the real backends must report unavailable
    instead of raising on import — and indexing must become a safe no-op.
    """

    def test_embedder_unavailable_raises_embeddingerror_only_on_use(self):
        from src.rag.embeddings import EmbeddingProvider, EmbeddingError
        provider = EmbeddingProvider()
        if provider.is_available():
            self.skipTest("sentence-transformers IS installed here")
        self.assertIsNotNone(provider.unavailable_reason)
        with self.assertRaises(EmbeddingError):
            provider.embed_text("hello")

    def test_store_unavailable_count_is_zero(self):
        from src.rag.vector_store import VectorStore
        store = VectorStore()
        if store.is_available():
            self.skipTest("chromadb IS installed here")
        self.assertEqual(store.count(), 0)

    def test_service_noops_when_unavailable(self):
        from src.rag.embeddings import EmbeddingProvider
        from src.rag.vector_store import VectorStore
        rag = RagService(db=DatabaseManager(db_path=tempfile.mktemp(suffix=".db")),
                         embedder=EmbeddingProvider(), store=VectorStore())
        if rag.is_available():
            self.skipTest("AI stack IS installed here")
        # No exceptions — safe no-ops.
        self.assertFalse(rag.index_email(1, "a@b.com", "s", "b"))
        self.assertEqual(rag.search("anything"), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
