#!/usr/bin/env bash
# =============================================================================
# Email Intelligence Platform — launcher
#
#   ./run.sh ui      Streamlit dashboard (default)
#   ./run.sh cli     Original interactive CLI (python main.py)
#   ./run.sh chat    Chat-with-emails REPL
#   ./run.sh index   Backfill embeddings for semantic search / chat
#   ./run.sh test    Run the test suite
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
MODE="${1:-ui}"

case "$MODE" in
    ui)
        echo "Launching Streamlit dashboard…"
        if command -v streamlit >/dev/null 2>&1; then
            streamlit run app.py
        else
            $PYTHON -m streamlit run app.py
        fi
        ;;
    cli)
        $PYTHON main.py
        ;;
    chat)
        $PYTHON scripts/chat.py
        ;;
    index)
        shift || true
        $PYTHON scripts/backfill_embeddings.py "$@"
        ;;
    test)
        $PYTHON tests/run_tests.py
        ;;
    *)
        echo "Usage: ./run.sh [ui|cli|chat|index|test]"
        exit 1
        ;;
esac
