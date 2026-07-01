"""LLM Settings — model selection, backend status, embedding backfill."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

import config
from src.ui.services import get_db, get_rag, get_llm_client, active_model
from src.ui.components import sidebar_model_selector

st.set_page_config(page_title="LLM Settings", page_icon="⚙️", layout="wide")
sidebar_model_selector()

st.title("⚙️ LLM Settings")

# ── Model selection ────────────────────────────────────────────────────────────
st.subheader("Model")
current = active_model()
options = list(config.SUPPORTED_LLM_MODELS)
if current not in options:
    options = [current] + options
choice = st.selectbox("Active LLM model", options, index=options.index(current))
st.session_state["llm_model"] = choice
st.caption("All local via Ollama — no data leaves your machine.")

# ── Backend status ─────────────────────────────────────────────────────────────
st.subheader("Status")
llm = get_llm_client(choice)
rag = get_rag()
db = get_db()

col1, col2 = st.columns(2)
with col1:
    st.markdown("**LLM (Ollama)**")
    if llm.is_available():
        st.success(f"Available · model `{choice}` ready")
        st.caption(f"Base URL: {config.OLLAMA_BASE_URL}")
        st.caption(f"Installed models: {', '.join(llm.list_models()) or '(none)'}")
    elif llm.daemon_reachable():
        # Daemon is up but the chosen model isn't pulled.
        st.warning(f"Daemon running, but model `{choice}` isn't pulled.")
        st.caption(f"Installed models: {', '.join(llm.list_models()) or '(none)'}")
        st.code(f"ollama pull {choice}", language="bash")
    else:
        st.error("Daemon unreachable")
        st.caption(llm.reason or "")
        st.code("ollama serve", language="bash")

with col2:
    st.markdown("**Embeddings / RAG**")
    if rag.is_available():
        st.success("Available")
        st.caption(f"Model: {config.EMBEDDING_MODEL}")
        st.caption(f"Vectors indexed: {rag.store.count()}")
        st.caption(f"Tracked in DB: {db.count_embeddings()}")
    else:
        st.error("Unavailable")
        st.caption(rag.unavailable_reason() or "")

# ── Cache & index maintenance ──────────────────────────────────────────────────
st.subheader("Maintenance")
st.caption(f"Cached LLM analyses: {db.count_llm_analyses()} · "
           f"Chat messages: {db.count_chat_messages()}")

if rag.is_available():
    colA, colB = st.columns(2)
    with colA:
        if st.button("Backfill embeddings"):
            with st.spinner("Embedding stored emails…"):
                stats = rag.backfill_from_db()
            st.success(f"Embedded {stats['embedded']}, "
                       f"skipped {stats['skipped']} (unchanged).")
    with colB:
        if st.button("Force re-embed all"):
            with st.spinner("Re-embedding…"):
                stats = rag.backfill_from_db(force=True)
            st.success(f"Re-embedded {stats['embedded']}.")
else:
    st.info("Install the RAG stack to enable embedding backfill.")

# ── Active configuration ───────────────────────────────────────────────────────
with st.expander("Active configuration"):
    st.code(config.summary())
