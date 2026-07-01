"""AI Summary — full LLM analysis of a single email."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import streamlit as st

from src.importance_predictor import ImportancePredictor
from src.ui.services import get_llm_service
from src.ui.components import (
    sidebar_model_selector, email_picker, render_unavailable,
)

st.set_page_config(page_title="AI Summary", page_icon="🧠", layout="wide")
sidebar_model_selector()

st.title("🧠 AI Summary")
st.caption("Summary · action items · deadlines · reply · category · entities · risk · ML explanation")

email = email_picker()
if email:
    with st.expander("Email body", expanded=False):
        st.text(email.get("email_body", ""))

    svc = get_llm_service()
    if not svc.is_available():
        render_unavailable(svc.unavailable_reason(), "AI Summary")
    elif st.button("Analyze with AI", type="primary"):
        eid = email["id"]
        # Recompute the importance breakdown so the LLM can explain the ML rating.
        predictor = ImportancePredictor()
        imp = predictor.calculate_importance(
            email_text=f"{email.get('subject', '')}\n{email.get('email_body', '')}",
            sender=email.get("sender", ""),
            sentiment=email.get("overall_sentiment", "neutral"),
        )

        with st.spinner("Running local LLM…"):
            summary = svc.summarize(email, eid)
            actions = svc.extract_action_items(email, eid)
            deadlines = svc.detect_deadlines(email, eid)
            reply = svc.suggest_reply(email, eid)
            category = svc.categorize(email, eid)
            entities = svc.extract_entities(email, eid)
            risk = svc.detect_risks(email, eid)
            explain = svc.explain_importance(
                email, imp.importance_level, imp.breakdown, eid)

        st.subheader("Summary")
        st.write(summary.get("summary", "—"))

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Action items")
            items = actions.get("action_items", [])
            st.markdown("\n".join(f"- {i}" for i in items) or "_None_")

            st.subheader("Deadlines")
            dls = deadlines.get("deadlines", [])
            if dls:
                for d in dls:
                    st.markdown(f"- **{d.get('text', '')}** → {d.get('when', '')}")
            else:
                st.markdown("_None_")

            st.subheader("Category")
            st.markdown(f"`{category.get('category', 'General')}`")
        with col2:
            st.subheader("Risk")
            level = risk.get("risk_level", "low")
            color = {"low": "🟢", "medium": "🟠", "high": "🔴"}.get(level, "⚪")
            st.markdown(f"{color} **{level.upper()}**")
            for r in risk.get("risks", []):
                st.markdown(f"- {r}")

            st.subheader("Entities")
            ents = entities.get("entities", {})
            for key, vals in ents.items():
                if vals:
                    st.markdown(f"**{key.title()}:** {', '.join(vals)}")

        st.subheader("Suggested reply")
        st.text_area("Reply", reply.get("reply", ""), height=160)

        st.subheader(f"Why the ML rated this {imp.importance_level}")
        st.info(explain.get("explanation", "—"))
