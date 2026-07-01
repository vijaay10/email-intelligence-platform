"""
Email Intelligence Platform — Streamlit Dashboard
=================================================
Home / overview page. The seven feature pages live in ``pages/`` and are picked
up automatically by Streamlit's multipage routing.

Run:
    streamlit run app.py

This UI is purely additive — the original CLI (`python main.py`) still works
exactly as before.
"""

import os
import sys

# Make the project importable when Streamlit runs this file directly.
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import streamlit as st

import config
from src.ui.services import get_db, get_rag, get_llm_client, active_model
from src.ui.components import sidebar_model_selector

st.set_page_config(page_title="Email Intelligence Platform",
                   page_icon="📧", layout="wide")

sidebar_model_selector()

st.title("📧 Email Intelligence Platform")
st.caption("Hybrid ML + LLM + RAG · local-only (Ollama) · TF-IDF + Naive Bayes preserved")

# ── System status ──────────────────────────────────────────────────────────────
db = get_db()
rag = get_rag()
llm = get_llm_client(active_model())

c1, c2, c3, c4 = st.columns(4)
stats = db.get_summary_stats()
c1.metric("Emails analysed", stats["total_emails"])
c2.metric("High importance", stats["high_importance"])
c3.metric("Sentiment shifts", stats["with_sentiment_shift"])
c4.metric("Avg urgency", f"{stats['avg_urgency_score']:.2f}")

st.divider()

s1, s2, s3 = st.columns(3)
with s1:
    st.subheader("LLM (Ollama)")
    if llm.is_available():
        st.success(f"Available · model `{active_model()}`")
    else:
        st.warning("Unavailable")
        st.caption(llm.reason or "")
with s2:
    st.subheader("Embeddings / RAG")
    if rag.is_available():
        st.success(f"Available · {rag.store.count()} vectors")
    else:
        st.warning("Unavailable")
        st.caption(rag.unavailable_reason() or "")
with s3:
    st.subheader("Database")
    st.success("Connected")
    st.caption(str(config.DB_PATH))

st.divider()

# ── Sentiment & importance snapshots (from the existing ML pipeline) ───────────
left, right = st.columns(2)
with left:
    st.subheader("Sentiment distribution")
    sentiment = db.get_sentiment_counts()
    if sentiment:
        st.bar_chart(sentiment)
    else:
        st.info("No data yet.")
with right:
    st.subheader("Importance distribution")
    importance = db.get_importance_counts()
    if importance:
        st.bar_chart(importance)
    else:
        st.info("No data yet.")

st.divider()
st.markdown(
    """
    ### Pages
    - **AI Summary** — LLM summary, action items, deadlines, reply, category, entities, risk, ML explanation
    - **Chat with Emails** — ask questions answered from your emails (RAG)
    - **Semantic Search** — find emails by meaning, not keywords
    - **Action Items** — tasks extracted across recent emails
    - **Important Emails** — high-importance emails with AI explanations
    - **Risk Dashboard** — risk flags across recent emails
    - **LLM Settings** — model selection, status, embedding backfill

    > Tip: index your emails for search/chat with `python scripts/backfill_embeddings.py`.
    """
)
