"""Important Emails — ML importance ranking with optional LLM explanations."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

from src.importance_predictor import ImportancePredictor
from src.ui.services import get_db, get_llm_service
from src.ui.components import sidebar_model_selector

st.set_page_config(page_title="Important Emails", page_icon="⭐", layout="wide")
sidebar_model_selector()

st.title("⭐ Important Emails")
st.caption("Ranked by the existing ML importance score (TF-IDF + Naive Bayes "
           "sentiment + 5-factor scoring). Ask the LLM why on demand.")

db = get_db()
level = st.radio("Importance level", ["HIGH", "MEDIUM", "LOW"], horizontal=True)
emails = db.get_emails_by_importance(level)

if not emails:
    st.info(f"No {level} importance emails yet.")
    st.stop()

st.caption(f"{len(emails)} {level} email(s).")
svc = get_llm_service()
predictor = ImportancePredictor()

for email in emails:
    with st.container(border=True):
        st.markdown(f"**{email.get('subject', '(no subject)')}**")
        st.caption(f"From {email.get('sender', '')} · "
                   f"sentiment {email.get('overall_sentiment', '')} · "
                   f"urgency {email.get('urgency_score', 0):.2f}")

        if svc.is_available():
            if st.button("Explain why", key=f"explain_{email['id']}"):
                imp = predictor.calculate_importance(
                    email_text=f"{email.get('subject', '')}\n{email.get('email_body', '')}",
                    sender=email.get("sender", ""),
                    sentiment=email.get("overall_sentiment", "neutral"),
                )
                with st.spinner("Asking the LLM…"):
                    res = svc.explain_importance(
                        email, imp.importance_level, imp.breakdown, email["id"])
                st.info(res.get("explanation", "—"))
        else:
            st.caption("LLM unavailable — explanations disabled.")
