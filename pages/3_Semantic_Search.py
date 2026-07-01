"""Semantic Search — find emails by meaning."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

import config
from src.ui.services import get_rag
from src.ui.components import sidebar_model_selector

st.set_page_config(page_title="Semantic Search", page_icon="🔍", layout="wide")
sidebar_model_selector()

st.title("🔍 Semantic Search")
st.caption("Search by meaning, not keywords. e.g. “overdue invoices”, "
           "“meetings tomorrow”, “AWS outages”.")

rag = get_rag()

if not rag.is_available():
    st.warning(f"Semantic search is unavailable — {rag.unavailable_reason()}")
    st.markdown(
        "**To enable it:**\n"
        "1. `pip install -r requirements.txt`\n"
        "2. `python scripts/backfill_embeddings.py` to index your emails"
    )
    st.stop()

st.caption(f"{rag.store.count()} emails indexed.")

query = st.text_input("Search query", placeholder="What invoices are overdue?")
top_k = st.slider("Results", 1, 20, config.RAG_TOP_K)

if query:
    with st.spinner("Searching…"):
        results = rag.search(query, top_k=top_k)

    if not results:
        st.info("No matches found.")
    for hit in results:
        meta = hit.get("metadata", {})
        score = hit.get("score")
        with st.container(border=True):
            st.markdown(f"**{meta.get('subject', '(no subject)')}**  ·  "
                        f"score `{score}`")
            st.caption(f"From {meta.get('sender', '')} · {meta.get('timestamp', '')}")
            st.write(hit.get("document", "")[:400])
