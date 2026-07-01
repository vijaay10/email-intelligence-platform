"""
Central Configuration  (single source of truth)
================================================
All paths, model names, and tunables for the Email Intelligence Platform
live here so they are no longer scattered across ``main.py``, the modules,
and the training scripts.

Design goals
------------
* **Stdlib-only.** Importing this module must never fail because an optional
  third-party package (pandas, ollama, chromadb, ...) is missing. It is the
  one module every other layer can safely import.
* **Environment-overridable.** Any value can be overridden via an environment
  variable or a ``.env`` file (see ``.env.example``) without code changes.
* **Backward compatible.** The default paths point at exactly the same
  locations the original project already used, so existing code keeps working.

Usage
-----
    import config
    print(config.LLM_MODEL)          # "llama3" (or whatever .env sets)
    model_path = config.MODEL_PATH
"""

from __future__ import annotations

import os
from pathlib import Path

# ── Project root ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent


# ── Minimal .env loader (no python-dotenv dependency) ──────────────────────────
def _load_env_file(path: Path) -> None:
    """
    Populate ``os.environ`` from a ``KEY=VALUE`` file if it exists.

    Existing environment variables always win (so real env vars override the
    file). Lines that are blank or start with ``#`` are ignored. Quotes around
    values are stripped. This is intentionally tiny — it covers the common case
    without pulling in a dependency.
    """
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)
    except OSError:
        # A malformed/unreadable .env should never crash the app.
        pass


# Load .env (root) and the legacy config/config.env, if present.
_load_env_file(ROOT / ".env")
_load_env_file(ROOT / "config" / "config.env")


# ── Typed environment helpers ──────────────────────────────────────────────────
def _env_str(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ── Filesystem layout (unchanged from the original project) ────────────────────
MODELS_DIR = ROOT / "models"
DB_DIR = ROOT / "db"
REPORTS_DIR = ROOT / "reports"
DATA_DIR = ROOT / "data"
CHROMA_DIR = Path(_env_str("CHROMA_DIR", str(ROOT / "db" / "chroma")))

MODEL_PATH = Path(_env_str("MODEL_PATH", str(MODELS_DIR / "sentiment_model.pkl")))
DB_PATH = Path(_env_str("DB_PATH", str(DB_DIR / "email_analysis.db")))
TRAIN_DATA_PATH = Path(_env_str("TRAIN_DATA_PATH", str(DATA_DIR / "train_data.csv")))

# ── Existing ML model tunables (match nlp_model.py defaults) ───────────────────
ML_MAX_FEATURES = _env_int("ML_MAX_FEATURES", 5000)
ML_NGRAM_MAX = _env_int("ML_NGRAM_MAX", 2)  # ngram_range = (1, ML_NGRAM_MAX)

# ── LLM layer (Ollama, local-only — no cloud APIs) ─────────────────────────────
LLM_PROVIDER = _env_str("LLM_PROVIDER", "ollama")
LLM_MODEL = _env_str("LLM_MODEL", "llama3")          # llama3 | gemma3 | mistral
OLLAMA_BASE_URL = _env_str("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_TIMEOUT = _env_int("LLM_TIMEOUT", 60)            # seconds per request
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_ENABLED = _env_bool("LLM_ENABLED", True)         # master kill-switch
LLM_CACHE_ENABLED = _env_bool("LLM_CACHE_ENABLED", True)

# Models the platform is validated against (for UI dropdowns / validation).
SUPPORTED_LLM_MODELS = ("llama3", "gemma3", "mistral")

# ── Embeddings / RAG ───────────────────────────────────────────────────────────
EMBEDDING_MODEL = _env_str("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBEDDING_BATCH_SIZE = _env_int("EMBEDDING_BATCH_SIZE", 32)
CHROMA_COLLECTION = _env_str("CHROMA_COLLECTION", "emails")
RAG_TOP_K = _env_int("RAG_TOP_K", 5)                 # neighbours per query

# ── Gmail (used by EmailFetcher / future ingestion) ────────────────────────────
GMAIL_EMAIL = _env_str("GMAIL_EMAIL", "")
GMAIL_APP_PASSWORD = _env_str("GMAIL_APP_PASSWORD", "")
IMAP_SERVER = _env_str("IMAP_SERVER", "imap.gmail.com")
MAX_EMAILS = _env_int("MAX_EMAILS", 10)

# ── Logging ────────────────────────────────────────────────────────────────────
LOG_LEVEL = _env_str("LOG_LEVEL", "WARNING")


def ensure_directories() -> None:
    """
    Create the runtime directories if they do not exist.

    Call this once at startup. It is a no-op when the directories already
    exist, and it never touches the source tree's existing data.
    """
    for directory in (MODELS_DIR, DB_DIR, REPORTS_DIR, DATA_DIR, CHROMA_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def summary() -> str:
    """Human-readable snapshot of the active configuration (for diagnostics)."""
    return "\n".join(
        [
            "Email Intelligence Platform — configuration",
            f"  ROOT            : {ROOT}",
            f"  MODEL_PATH      : {MODEL_PATH}",
            f"  DB_PATH         : {DB_PATH}",
            f"  CHROMA_DIR      : {CHROMA_DIR}",
            f"  LLM_PROVIDER    : {LLM_PROVIDER}",
            f"  LLM_MODEL       : {LLM_MODEL}",
            f"  OLLAMA_BASE_URL : {OLLAMA_BASE_URL}",
            f"  LLM_ENABLED     : {LLM_ENABLED}",
            f"  EMBEDDING_MODEL : {EMBEDDING_MODEL}",
            f"  CHROMA_COLLECTION: {CHROMA_COLLECTION}",
            f"  RAG_TOP_K       : {RAG_TOP_K}",
        ]
    )


if __name__ == "__main__":
    print(summary())
