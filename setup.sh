#!/usr/bin/env bash
# =============================================================================
# Email Intelligence Platform — setup
# Installs dependencies, prepares NLTK data, trains the ML model if needed,
# and (optionally) pulls the local LLM model. Optional steps warn instead of
# failing, so the core pipeline is usable even without the AI stack.
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
LLM_MODEL="${LLM_MODEL:-llama3}"

echo "=================================================================="
echo "  Email Intelligence Platform — setup"
echo "=================================================================="

# 1. Python deps -------------------------------------------------------------
echo "[1/5] Installing Python dependencies…"
$PYTHON -m pip install --upgrade pip
if ! $PYTHON -m pip install -r requirements.txt; then
    echo "  [!] Some packages failed to install. The core ML pipeline may still"
    echo "      work; AI features (LLM/RAG) need their packages. Continuing…"
fi

# 2. .env --------------------------------------------------------------------
echo "[2/5] Preparing .env…"
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    echo "  Created .env from .env.example (edit it to add Gmail / model settings)."
else
    echo "  .env already present or no template — skipping."
fi

# 3. NLTK data ---------------------------------------------------------------
echo "[3/5] Downloading NLTK data (punkt, stopwords)…"
$PYTHON - <<'PY' || echo "  [!] NLTK download skipped (will fall back to built-ins)."
import nltk
for pkg in ("punkt", "punkt_tab", "stopwords"):
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass
PY

# 4. Train ML model if missing ----------------------------------------------
echo "[4/5] Ensuring the sentiment model exists…"
if [ ! -f models/sentiment_model.pkl ]; then
    echo "  Training model…"
    $PYTHON scripts/train_model.py || echo "  [!] Training failed (need scikit-learn)."
else
    echo "  Model already trained — skipping."
fi

# 5. Ollama model (optional) -------------------------------------------------
echo "[5/5] Local LLM (Ollama)…"
if command -v ollama >/dev/null 2>&1; then
    echo "  Pulling model '$LLM_MODEL' (Ctrl+C to skip)…"
    ollama pull "$LLM_MODEL" || echo "  [!] Could not pull '$LLM_MODEL'."
else
    echo "  Ollama not found. LLM features will be disabled until you install it:"
    echo "      https://ollama.com  →  then: ollama pull $LLM_MODEL"
fi

echo "=================================================================="
echo "  Setup complete.  Next:  ./run.sh ui    (or)    ./run.sh cli"
echo "=================================================================="
