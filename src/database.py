"""
Database Layer  (Feature 3)
----------------------------
SQLite-backed storage for email analysis results and sentence-level
sentiment data.  Uses the standard library ``sqlite3`` — no ORM
dependency needed, keeping the project lightweight.

Schema
------
emails(id, sender, subject, email_body, overall_sentiment,
       importance_level, sentiment_shift, urgency_score, date,
       created_at)

email_sentences(id, email_id, sentence_index, sentence,
                sentiment, confidence)
"""

import sqlite3
import os
import logging
from contextlib import contextmanager
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── SQL DDL ───────────────────────────────────────────────────────────────────

# NOTE: This module is the single source of truth for the database schema.
# db/schema.sql is a human-readable mirror of the DDL below — keep them in sync.

CREATE_EMAILS_TABLE = """
CREATE TABLE IF NOT EXISTS emails (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    sender           TEXT    NOT NULL,
    subject          TEXT,
    email_body       TEXT,
    overall_sentiment TEXT   NOT NULL
                      CHECK (overall_sentiment IN ('positive','neutral','negative')),
    importance_level  TEXT   NOT NULL
                      CHECK (importance_level  IN ('LOW','MEDIUM','HIGH')),
    sentiment_shift   INTEGER NOT NULL DEFAULT 0,   -- BOOLEAN (0/1)
    urgency_score     REAL    NOT NULL DEFAULT 0.0,
    date             TEXT,
    created_at       TEXT    NOT NULL
);
"""

CREATE_SENTENCES_TABLE = """
CREATE TABLE IF NOT EXISTS email_sentences (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id       INTEGER NOT NULL REFERENCES emails(id) ON DELETE CASCADE,
    sentence_index INTEGER NOT NULL,
    sentence       TEXT    NOT NULL,
    sentiment      TEXT    NOT NULL
                   CHECK (sentiment IN ('positive','neutral','negative')),
    confidence     REAL    NOT NULL DEFAULT 0.0
);
"""

# ── email_embeddings (Phase 1 — RAG) ──────────────────────────────────────────
# Tracks which emails have been embedded into the ChromaDB vector store. The
# vectors themselves live in Chroma; this table records the content hash so we
# can skip re-embedding emails whose text has not changed.
CREATE_EMBEDDINGS_TABLE = """
CREATE TABLE IF NOT EXISTS email_embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id        INTEGER NOT NULL UNIQUE
                    REFERENCES emails(id) ON DELETE CASCADE,
    chroma_id       TEXT    NOT NULL UNIQUE,
    content_hash    TEXT    NOT NULL,
    embedding_model TEXT    NOT NULL,
    vector_dim      INTEGER,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
"""

CREATE_IDX_EMBEDDINGS_HASH = """
CREATE INDEX IF NOT EXISTS idx_embeddings_hash
    ON email_embeddings(content_hash);
"""

# ── llm_analysis (Phase 2 — LLM cache) ─────────────────────────────────────────
# Caches LLM feature outputs (summary, action_items, ...) keyed by a hash of
# (feature + model + input text), so we never call the LLM twice for the same
# input. ``result`` holds the JSON-encoded feature output.
CREATE_LLM_ANALYSIS_TABLE = """
CREATE TABLE IF NOT EXISTS llm_analysis (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email_id    INTEGER,                 -- optional link to emails(id); not a FK
                                         -- so the cache also works for transient
                                         -- (not-yet-persisted) emails
    feature     TEXT    NOT NULL,
    model       TEXT    NOT NULL,
    cache_key   TEXT    NOT NULL UNIQUE,
    result      TEXT    NOT NULL,
    created_at  TEXT    NOT NULL
);
"""

CREATE_IDX_LLM_EMAIL_FEATURE = """
CREATE INDEX IF NOT EXISTS idx_llm_email_feature
    ON llm_analysis(email_id, feature);
"""

# ── chat_history / conversation_memory (Phase 3 — Chat with Emails) ─────────────
CREATE_CHAT_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS chat_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL,
    role            TEXT    NOT NULL
                    CHECK (role IN ('user','assistant','system')),
    content         TEXT    NOT NULL,
    sources         TEXT,                -- JSON list of retrieved email refs
    created_at      TEXT    NOT NULL
);
"""

CREATE_CONVERSATION_MEMORY_TABLE = """
CREATE TABLE IF NOT EXISTS conversation_memory (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT    NOT NULL UNIQUE,
    summary         TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL
);
"""

CREATE_IDX_CHAT_CONVERSATION = """
CREATE INDEX IF NOT EXISTS idx_chat_conversation
    ON chat_history(conversation_id);
"""

CREATE_IDX_EMAIL_SENTIMENT = """
CREATE INDEX IF NOT EXISTS idx_emails_sentiment
    ON emails(overall_sentiment);
"""

CREATE_IDX_EMAIL_IMPORTANCE = """
CREATE INDEX IF NOT EXISTS idx_emails_importance
    ON emails(importance_level);
"""

CREATE_IDX_EMAIL_DATE = """
CREATE INDEX IF NOT EXISTS idx_emails_date
    ON emails(created_at);
"""

CREATE_IDX_EMAIL_SENDER = """
CREATE INDEX IF NOT EXISTS idx_emails_sender
    ON emails(sender);
"""

CREATE_IDX_SENTENCES_EMAIL = """
CREATE INDEX IF NOT EXISTS idx_sentences_email_id
    ON email_sentences(email_id);
"""

# ── Analytical views (mirror db/schema.sql — used by BI tools and analytics) ───
CREATE_VIEW_DAILY_SENTIMENT = """
CREATE VIEW IF NOT EXISTS v_daily_sentiment AS
    SELECT DATE(created_at) AS day, overall_sentiment, COUNT(*) AS email_count
    FROM emails
    GROUP BY day, overall_sentiment
    ORDER BY day;
"""

CREATE_VIEW_NEGATIVE_SENDERS = """
CREATE VIEW IF NOT EXISTS v_negative_senders AS
    SELECT sender, COUNT(*) AS negative_count
    FROM emails
    WHERE overall_sentiment = 'negative'
    GROUP BY sender
    ORDER BY negative_count DESC;
"""

CREATE_VIEW_HIGH_IMPORTANCE = """
CREATE VIEW IF NOT EXISTS v_high_importance AS
    SELECT id, sender, SUBSTR(subject, 1, 60) AS subject_preview,
           overall_sentiment, urgency_score, date, created_at
    FROM emails
    WHERE importance_level = 'HIGH'
    ORDER BY urgency_score DESC;
"""

CREATE_VIEW_SHIFT_SUMMARY = """
CREATE VIEW IF NOT EXISTS v_shift_summary AS
    SELECT DATE(created_at) AS day, SUM(sentiment_shift) AS shifts_detected,
           COUNT(*) AS total_emails
    FROM emails
    GROUP BY day
    ORDER BY day;
"""


# ── DatabaseManager ───────────────────────────────────────────────────────────

class DatabaseManager:
    """
    Thread-safe SQLite wrapper.

    Parameters
    ----------
    db_path : str
        Path to the SQLite database file.
        Defaults to ``db/email_analysis.db`` relative to the project root.
    """

    DEFAULT_DB_PATH = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "db", "email_analysis.db"
    )

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self.DEFAULT_DB_PATH
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._initialise()

    # ── Internal helpers ──────────────────────────────────────────────────

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row          # access columns by name
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")  # concurrent reads
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _initialise(self) -> None:
        """Create tables and indexes if they do not exist."""
        with self._get_connection() as conn:
            for ddl in [
                CREATE_EMAILS_TABLE,
                CREATE_SENTENCES_TABLE,
                CREATE_EMBEDDINGS_TABLE,
                CREATE_LLM_ANALYSIS_TABLE,
                CREATE_CHAT_HISTORY_TABLE,
                CREATE_CONVERSATION_MEMORY_TABLE,
                CREATE_IDX_EMAIL_SENTIMENT,
                CREATE_IDX_EMAIL_IMPORTANCE,
                CREATE_IDX_EMAIL_DATE,
                CREATE_IDX_EMAIL_SENDER,
                CREATE_IDX_SENTENCES_EMAIL,
                CREATE_IDX_EMBEDDINGS_HASH,
                CREATE_IDX_LLM_EMAIL_FEATURE,
                CREATE_IDX_CHAT_CONVERSATION,
                CREATE_VIEW_DAILY_SENTIMENT,
                CREATE_VIEW_NEGATIVE_SENDERS,
                CREATE_VIEW_HIGH_IMPORTANCE,
                CREATE_VIEW_SHIFT_SUMMARY,
            ]:
                conn.execute(ddl)
        logger.info("[DB] Initialised at %s", self.db_path)

    # ── Write operations ──────────────────────────────────────────────────

    def save_email_analysis(
        self,
        sender: str,
        subject: str,
        email_body: str,
        overall_sentiment: str,
        importance_level: str,
        sentiment_shift: bool,
        urgency_score: float,
        date: str,
        sentences: List[Dict],          # [{sentence, sentiment, confidence, index}]
    ) -> int:
        """
        Persist a complete email analysis result.

        Returns
        -------
        int
            The auto-generated email row ``id``.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO emails
                    (sender, subject, email_body, overall_sentiment,
                     importance_level, sentiment_shift, urgency_score,
                     date, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sender, subject, email_body, overall_sentiment,
                    importance_level, int(sentiment_shift), urgency_score,
                    date, now,
                ),
            )
            email_id = cursor.lastrowid

            if sentences:
                conn.executemany(
                    """
                    INSERT INTO email_sentences
                        (email_id, sentence_index, sentence, sentiment, confidence)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            email_id,
                            s.get("index", idx + 1),
                            s["sentence"],
                            s["sentiment"],
                            s.get("confidence", 0.0),
                        )
                        for idx, s in enumerate(sentences)
                    ],
                )

        logger.info("[DB] Saved email_id=%d | sentiment=%s | importance=%s",
                    email_id, overall_sentiment, importance_level)
        return email_id

    # ── Embedding records (Phase 1 — RAG) ─────────────────────────────────

    def save_embedding_record(
        self,
        email_id: int,
        chroma_id: str,
        content_hash: str,
        embedding_model: str,
        vector_dim: Optional[int] = None,
    ) -> None:
        """
        Upsert the embedding-tracking record for an email.

        Uses ``ON CONFLICT(email_id)`` so re-embedding an email updates the
        existing row (new hash / model) rather than creating a duplicate.
        """
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO email_embeddings
                    (email_id, chroma_id, content_hash, embedding_model,
                     vector_dim, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(email_id) DO UPDATE SET
                    chroma_id       = excluded.chroma_id,
                    content_hash    = excluded.content_hash,
                    embedding_model = excluded.embedding_model,
                    vector_dim      = excluded.vector_dim,
                    updated_at      = excluded.updated_at
                """,
                (email_id, chroma_id, content_hash, embedding_model,
                 vector_dim, now, now),
            )
        logger.info("[DB] Embedding record saved for email_id=%d", email_id)

    def get_embedding_record(self, email_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM email_embeddings WHERE email_id = ?", (email_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_embedded_email_ids(self) -> set:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT email_id FROM email_embeddings"
            ).fetchall()
        return {r["email_id"] for r in rows}

    def count_embeddings(self) -> int:
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM email_embeddings"
            ).fetchone()[0]

    # ── LLM analysis cache (Phase 2) ──────────────────────────────────────

    def get_llm_cache(self, cache_key: str) -> Optional[Dict]:
        """Return a cached LLM result row by cache key, or None."""
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM llm_analysis WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return dict(row) if row else None

    def save_llm_cache(
        self,
        cache_key: str,
        feature: str,
        model: str,
        result: str,
        email_id: Optional[int] = None,
    ) -> None:
        """Upsert a cached LLM result (``result`` is a JSON string)."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO llm_analysis
                    (email_id, feature, model, cache_key, result, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    result     = excluded.result,
                    model      = excluded.model,
                    created_at = excluded.created_at
                """,
                (email_id, feature, model, cache_key, result, now),
            )

    def get_llm_results_for_email(self, email_id: int) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM llm_analysis WHERE email_id = ?", (email_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def count_llm_analyses(self) -> int:
        with self._get_connection() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM llm_analysis"
            ).fetchone()[0]

    # ── Chat history & conversation memory (Phase 3) ──────────────────────

    def save_chat_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[str] = None,
    ) -> int:
        """Append one chat message. ``sources`` is an optional JSON string."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_history
                    (conversation_id, role, content, sources, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, sources, now),
            )
            return cursor.lastrowid

    def get_chat_history(self, conversation_id: str) -> List[Dict]:
        """All messages for a conversation, oldest first."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM chat_history WHERE conversation_id = ? ORDER BY id",
                (conversation_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_conversations(self) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT conversation_id,
                       COUNT(*)        AS message_count,
                       MAX(created_at) AS last_at
                FROM chat_history
                GROUP BY conversation_id
                ORDER BY last_at DESC
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def count_chat_messages(self) -> int:
        with self._get_connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]

    def delete_conversation(self, conversation_id: str) -> None:
        """Remove a conversation's messages and its rolling memory."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM chat_history WHERE conversation_id = ?",
                         (conversation_id,))
            conn.execute("DELETE FROM conversation_memory WHERE conversation_id = ?",
                         (conversation_id,))
        logger.info("[DB] Cleared conversation '%s'", conversation_id)

    def get_conversation_memory(self, conversation_id: str) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM conversation_memory WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
        return dict(row) if row else None

    def save_conversation_memory(self, conversation_id: str, summary: str) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO conversation_memory (conversation_id, summary, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(conversation_id) DO UPDATE SET
                    summary    = excluded.summary,
                    updated_at = excluded.updated_at
                """,
                (conversation_id, summary, now),
            )

    # ── Read operations ───────────────────────────────────────────────────

    def get_all_emails(self) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails ORDER BY created_at DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_email_by_id(self, email_id: int) -> Optional[Dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM emails WHERE id = ?", (email_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_sentences_for_email(self, email_id: int) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM email_sentences
                WHERE email_id = ?
                ORDER BY sentence_index
                """,
                (email_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_emails_by_sentiment(self, sentiment: str) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails WHERE overall_sentiment = ? ORDER BY created_at DESC",
                (sentiment,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_emails_by_importance(self, level: str) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM emails WHERE importance_level = ? ORDER BY created_at DESC",
                (level,),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Analytics queries ─────────────────────────────────────────────────

    def get_sentiment_counts(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT overall_sentiment, COUNT(*) AS cnt
                FROM emails
                GROUP BY overall_sentiment
                """
            ).fetchall()
        return {r["overall_sentiment"]: r["cnt"] for r in rows}

    def get_importance_counts(self) -> Dict[str, int]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT importance_level, COUNT(*) AS cnt
                FROM emails
                GROUP BY importance_level
                """
            ).fetchall()
        return {r["importance_level"]: r["cnt"] for r in rows}

    def get_sentiment_trend(self) -> List[Dict]:
        """Daily sentiment counts for trend charts."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    DATE(created_at) AS day,
                    overall_sentiment,
                    COUNT(*) AS cnt
                FROM emails
                GROUP BY day, overall_sentiment
                ORDER BY day
                """
            ).fetchall()
        return [dict(r) for r in rows]

    def get_most_negative_senders(self, top_n: int = 5) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT sender, COUNT(*) AS negative_count
                FROM emails
                WHERE overall_sentiment = 'negative'
                GROUP BY sender
                ORDER BY negative_count DESC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_most_urgent_emails(self, top_n: int = 10) -> List[Dict]:
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, sender, subject, urgency_score,
                       importance_level, overall_sentiment, date
                FROM emails
                ORDER BY urgency_score DESC
                LIMIT ?
                """,
                (top_n,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_summary_stats(self) -> Dict:
        with self._get_connection() as conn:
            total = conn.execute("SELECT COUNT(*) FROM emails").fetchone()[0]
            high = conn.execute(
                "SELECT COUNT(*) FROM emails WHERE importance_level = 'HIGH'"
            ).fetchone()[0]
            shift = conn.execute(
                "SELECT COUNT(*) FROM emails WHERE sentiment_shift = 1"
            ).fetchone()[0]
            avg_urgency = conn.execute(
                "SELECT AVG(urgency_score) FROM emails"
            ).fetchone()[0] or 0.0

        return {
            "total_emails": total,
            "high_importance": high,
            "with_sentiment_shift": shift,
            "avg_urgency_score": round(avg_urgency, 2),
        }

    # ── Utility ───────────────────────────────────────────────────────────

    def delete_email(self, email_id: int) -> None:
        """Hard-delete an email and its sentences (cascade)."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM emails WHERE id = ?", (email_id,))
        logger.info("[DB] Deleted email_id=%d", email_id)

    def reset_database(self) -> None:
        """Drop all rows — useful for testing."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM email_sentences")
            conn.execute("DELETE FROM emails")
        logger.warning("[DB] Database reset — all records deleted")
