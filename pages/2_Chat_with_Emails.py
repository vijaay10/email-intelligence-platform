"""Chat with Emails — RAG-grounded assistant."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

from src.ui.services import get_chat_service
from src.ui.components import (
    sidebar_model_selector, render_sources, render_unavailable,
)

st.set_page_config(page_title="Chat with Emails", page_icon="💬", layout="wide")
sidebar_model_selector()

st.title("💬 Chat with Emails")
st.caption("Ask questions answered from your indexed emails. e.g. "
           "“What invoices are pending?” · “Which emails mention Kubernetes?”")

chat = get_chat_service()

if not chat.is_available():
    render_unavailable(chat.unavailable_reason(), "Chat")
    st.stop()

if not chat.rag.is_available():
    st.info("Semantic search is unavailable, so answers won't be grounded in your "
            "emails yet. Install the RAG stack and run "
            "`python scripts/backfill_embeddings.py`.")

CONV = "streamlit"

# Replay history.
for msg in chat.history(CONV):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask about your emails…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = chat.ask(prompt, conversation_id=CONV)
        st.markdown(result["answer"] or "_(no answer)_")
        render_sources(result.get("sources", []))

if st.button("Clear chat"):
    chat.db.delete_conversation(CONV)
    st.rerun()
