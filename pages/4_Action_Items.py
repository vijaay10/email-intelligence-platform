"""Action Items — tasks extracted across recent emails."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

from src.ui.services import get_db, get_llm_service
from src.ui.components import sidebar_model_selector, render_unavailable

st.set_page_config(page_title="Action Items", page_icon="✅", layout="wide")
sidebar_model_selector()

st.title("✅ Action Items")
st.caption("Tasks the LLM extracted from your recent emails (cached after first run).")

svc = get_llm_service()
if not svc.is_available():
    render_unavailable(svc.unavailable_reason(), "Action item extraction")
    st.stop()

emails = get_db().get_all_emails()
if not emails:
    st.info("No emails yet. Run `python main.py` → option 2 first.")
    st.stop()

limit = st.slider("How many recent emails to scan", 1, min(50, len(emails)),
                  min(10, len(emails)))

if st.button("Extract action items", type="primary"):
    found_any = False
    with st.spinner("Extracting…"):
        for email in emails[:limit]:
            result = svc.extract_action_items(email, email["id"])
            items = result.get("action_items", [])
            if items:
                found_any = True
                with st.container(border=True):
                    st.markdown(f"**{email.get('subject', '(no subject)')}** "
                                f"— {email.get('sender', '')}")
                    for item in items:
                        st.checkbox(item, key=f"ai_{email['id']}_{item[:30]}")
    if not found_any:
        st.info("No action items found in the scanned emails.")
