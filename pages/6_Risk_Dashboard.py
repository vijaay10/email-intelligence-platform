"""Risk Dashboard — LLM-detected risk flags across recent emails."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

from src.ui.services import get_db, get_llm_service
from src.ui.components import sidebar_model_selector, render_unavailable

st.set_page_config(page_title="Risk Dashboard", page_icon="⚠️", layout="wide")
sidebar_model_selector()

st.title("⚠️ Risk Dashboard")
st.caption("Risk assessment (priority · client escalation · security · legal · "
           "financial) across recent emails.")

svc = get_llm_service()
if not svc.is_available():
    render_unavailable(svc.unavailable_reason(), "Risk detection")
    st.stop()

emails = get_db().get_all_emails()
if not emails:
    st.info("No emails yet. Run `python main.py` → option 2 first.")
    st.stop()

limit = st.slider("How many recent emails to scan", 1, min(50, len(emails)),
                  min(10, len(emails)))

if st.button("Assess risk", type="primary"):
    counts = {"high": 0, "medium": 0, "low": 0}
    flagged = []
    with st.spinner("Assessing…"):
        for email in emails[:limit]:
            res = svc.detect_risks(email, email["id"])
            level = res.get("risk_level", "low")
            counts[level] = counts.get(level, 0) + 1
            if level in ("high", "medium"):
                flagged.append((email, res))

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High", counts["high"])
    c2.metric("🟠 Medium", counts["medium"])
    c3.metric("🟢 Low", counts["low"])
    st.bar_chart(counts)

    st.subheader("Flagged emails")
    if not flagged:
        st.success("No elevated-risk emails in the scanned set.")
    for email, res in flagged:
        icon = "🔴" if res["risk_level"] == "high" else "🟠"
        with st.container(border=True):
            st.markdown(f"{icon} **{email.get('subject', '(no subject)')}** "
                        f"— {email.get('sender', '')}")
            for r in res.get("risks", []):
                st.markdown(f"- {r}")
