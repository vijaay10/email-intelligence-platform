"""Fetch Emails — pull a real Gmail inbox, analyze, and index it."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

import config
from src.email_fetcher import EmailFetcher
from src.ui.services import get_analyzer, get_rag, get_db
from src.ui.components import sidebar_model_selector

st.set_page_config(page_title="Fetch Emails", page_icon="📥", layout="wide")
sidebar_model_selector()

st.title("📥 Fetch Emails")
st.caption("Pull your real Gmail inbox, run the ML pipeline, and index for search/chat.")

with st.expander("How to get a Gmail App Password", expanded=False):
    st.markdown(
        "1. Google Account → **Security** → enable **2-Step Verification**\n"
        "2. Search **App Passwords** → create one for *Mail*\n"
        "3. Use that 16-character password below (not your normal password).\n\n"
        "🔒 Credentials are used only for this IMAP session and are never stored."
    )

# Pre-fill from .env / config if present.
default_email = config.GMAIL_EMAIL or ""
col1, col2 = st.columns(2)
with col1:
    email_addr = st.text_input("Gmail address", value=default_email)
with col2:
    app_password = st.text_input("App password", type="password",
                                 value=config.GMAIL_APP_PASSWORD or "")

c1, c2, c3 = st.columns(3)
with c1:
    limit = st.number_input("How many emails", 1, 200, config.MAX_EMAILS)
with c2:
    unread_only = st.checkbox("Unread only", value=False)
with c3:
    auto_index = st.checkbox("Index for search/chat", value=True)

if st.button("Fetch & analyze", type="primary", disabled=not (email_addr and app_password)):
    fetcher = EmailFetcher(email_addr, app_password, imap_server=config.IMAP_SERVER)
    with st.spinner("Connecting to Gmail…"):
        connected = fetcher.connect()
    if not connected:
        st.error("Could not connect. Check the address and App Password (not your "
                 "normal password), and that IMAP is enabled in Gmail settings.")
    else:
        with st.spinner(f"Fetching {int(limit)} email(s)…"):
            emails = (fetcher.get_unread_emails(limit=int(limit)) if unread_only
                      else fetcher.get_inbox_emails(limit=int(limit)))
            fetcher.disconnect()

        if not emails:
            st.warning("No emails found for that filter.")
        else:
            with st.spinner(f"Analyzing {len(emails)} email(s) with the ML pipeline…"):
                results = get_analyzer().analyse_emails(emails)
            st.success(f"Analyzed and stored {len(results)} email(s).")

            rows = [{
                "subject": (r.get("subject") or "")[:60],
                "sender": r.get("sender", ""),
                "sentiment": r.get("overall_sentiment", ""),
                "importance": r.get("importance_level", ""),
                "shift": "⚠" if r.get("sentiment_shift") else "",
            } for r in results]
            st.dataframe(rows, use_container_width=True)

            if auto_index:
                rag = get_rag()
                if rag.is_available():
                    with st.spinner("Indexing embeddings for semantic search / chat…"):
                        stats = rag.backfill_from_db()
                    st.success(f"Indexed {stats['embedded']} new "
                               f"(skipped {stats['skipped']} unchanged). "
                               f"Total vectors: {rag.store.count()}.")
                else:
                    st.info(f"Skipped indexing — RAG unavailable "
                            f"({rag.unavailable_reason()}).")

            st.caption(f"Total emails in database: {get_db().get_summary_stats()['total_emails']}")
