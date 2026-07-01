-- ============================================================
-- Advanced Intelligent Email Analysis System
-- Database Schema — SQLite
-- ============================================================
-- NOTE: This file is a human-readable MIRROR of the runtime DDL in
-- src/database.py (DatabaseManager._initialise), which is the single
-- source of truth. The application builds the schema from there; keep
-- the two in sync when adding tables/indexes/views.
-- ============================================================

-- Enable foreign key enforcement (run once per connection)
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

-- ------------------------------------------------------------
-- Table: emails
-- Stores one row per analysed email with all top-level results
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS emails (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    sender            TEXT     NOT NULL,
    subject           TEXT,
    email_body        TEXT,
    overall_sentiment TEXT     NOT NULL
                      CHECK (overall_sentiment IN ('positive','neutral','negative')),
    importance_level  TEXT     NOT NULL
                      CHECK (importance_level  IN ('LOW','MEDIUM','HIGH')),
    sentiment_shift   INTEGER  NOT NULL DEFAULT 0,   -- BOOLEAN: 0=False, 1=True
    urgency_score     REAL     NOT NULL DEFAULT 0.0,
    date              TEXT,                           -- original email date header
    created_at        TEXT     NOT NULL               -- UTC ISO-8601 timestamp
);

-- ------------------------------------------------------------
-- Table: email_sentences
-- Stores sentence-level sentiment for each email (Feature 1)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS email_sentences (
    id             INTEGER  PRIMARY KEY AUTOINCREMENT,
    email_id       INTEGER  NOT NULL
                   REFERENCES emails(id) ON DELETE CASCADE,
    sentence_index INTEGER  NOT NULL,
    sentence       TEXT     NOT NULL,
    sentiment      TEXT     NOT NULL
                   CHECK (sentiment IN ('positive','neutral','negative')),
    confidence     REAL     NOT NULL DEFAULT 0.0
);

-- ------------------------------------------------------------
-- Table: email_embeddings  (Phase 1 — RAG)
-- Tracks which emails have been embedded into the ChromaDB vector
-- store. Vectors live in Chroma; the content hash here lets the app
-- skip re-embedding emails whose text has not changed.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS email_embeddings (
    id              INTEGER  PRIMARY KEY AUTOINCREMENT,
    email_id        INTEGER  NOT NULL UNIQUE
                    REFERENCES emails(id) ON DELETE CASCADE,
    chroma_id       TEXT     NOT NULL UNIQUE,
    content_hash    TEXT     NOT NULL,
    embedding_model TEXT     NOT NULL,
    vector_dim      INTEGER,
    created_at      TEXT     NOT NULL,
    updated_at      TEXT     NOT NULL
);

-- ------------------------------------------------------------
-- Table: llm_analysis  (Phase 2 — LLM cache)
-- Caches LLM feature outputs keyed by a hash of (feature + model +
-- input text) so the local LLM is never called twice for the same
-- input. 'result' holds the JSON-encoded feature output.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS llm_analysis (
    id          INTEGER  PRIMARY KEY AUTOINCREMENT,
    email_id    INTEGER,                  -- optional link to emails(id), not a FK
    feature     TEXT     NOT NULL,
    model       TEXT     NOT NULL,
    cache_key   TEXT     NOT NULL UNIQUE,
    result      TEXT     NOT NULL,
    created_at  TEXT     NOT NULL
);

-- ------------------------------------------------------------
-- Tables: chat_history / conversation_memory  (Phase 3 — Chat)
-- chat_history stores every chat turn; conversation_memory holds a
-- rolling summary so long conversations stay within the LLM context.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS chat_history (
    id              INTEGER  PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT     NOT NULL,
    role            TEXT     NOT NULL
                    CHECK (role IN ('user','assistant','system')),
    content         TEXT     NOT NULL,
    sources         TEXT,                 -- JSON list of retrieved email refs
    created_at      TEXT     NOT NULL
);

CREATE TABLE IF NOT EXISTS conversation_memory (
    id              INTEGER  PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT     NOT NULL UNIQUE,
    summary         TEXT     NOT NULL,
    updated_at      TEXT     NOT NULL
);

-- ------------------------------------------------------------
-- Indexes for common query patterns
-- ------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_embeddings_hash
    ON email_embeddings (content_hash);

CREATE INDEX IF NOT EXISTS idx_llm_email_feature
    ON llm_analysis (email_id, feature);

CREATE INDEX IF NOT EXISTS idx_chat_conversation
    ON chat_history (conversation_id);

CREATE INDEX IF NOT EXISTS idx_emails_sentiment
    ON emails (overall_sentiment);

CREATE INDEX IF NOT EXISTS idx_emails_importance
    ON emails (importance_level);

CREATE INDEX IF NOT EXISTS idx_emails_date
    ON emails (created_at);

CREATE INDEX IF NOT EXISTS idx_emails_sender
    ON emails (sender);

CREATE INDEX IF NOT EXISTS idx_sentences_email_id
    ON email_sentences (email_id);

-- ============================================================
-- Useful analytical views (optional, for external BI tools)
-- ============================================================

-- Daily sentiment counts
CREATE VIEW IF NOT EXISTS v_daily_sentiment AS
    SELECT
        DATE(created_at)  AS day,
        overall_sentiment,
        COUNT(*)          AS email_count
    FROM emails
    GROUP BY day, overall_sentiment
    ORDER BY day;

-- Most negative senders
CREATE VIEW IF NOT EXISTS v_negative_senders AS
    SELECT
        sender,
        COUNT(*) AS negative_count
    FROM emails
    WHERE overall_sentiment = 'negative'
    GROUP BY sender
    ORDER BY negative_count DESC;

-- High importance emails with subject preview
CREATE VIEW IF NOT EXISTS v_high_importance AS
    SELECT
        id,
        sender,
        SUBSTR(subject, 1, 60) AS subject_preview,
        overall_sentiment,
        urgency_score,
        date,
        created_at
    FROM emails
    WHERE importance_level = 'HIGH'
    ORDER BY urgency_score DESC;

-- Shift summary
CREATE VIEW IF NOT EXISTS v_shift_summary AS
    SELECT
        DATE(created_at)    AS day,
        SUM(sentiment_shift) AS shifts_detected,
        COUNT(*)             AS total_emails
    FROM emails
    GROUP BY day
    ORDER BY day;
