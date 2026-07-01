"""
Shared Streamlit render components.
===================================
Small helpers reused across pages: the email picker, source rendering, the
sidebar model selector, and availability badges. Keeping these here avoids
copy-paste across the seven pages.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import streamlit as st

import config
from src.ui.services import get_db


def sidebar_model_selector() -> str:
    """Render the model dropdown in the sidebar; returns the active model."""
    current = st.session_state.get("llm_model", config.LLM_MODEL)
    options = list(config.SUPPORTED_LLM_MODELS)
    if current not in options:
        options = [current] + options
    choice = st.sidebar.selectbox(
        "LLM model", options, index=options.index(current),
        help="Local Ollama model. Switch on the LLM Settings page too.",
    )
    st.session_state["llm_model"] = choice
    return choice


def availability_badge(available: bool, reason: Optional[str] = None,
                       label: str = "LLM") -> None:
    if available:
        st.success(f"{label}: available")
    else:
        st.warning(f"{label}: unavailable — {reason or 'not reachable'}")


def email_picker(key: str = "email_picker") -> Optional[Dict]:
    """
    Render a selectbox of stored emails. Returns the selected email row
    (a dict with id/sender/subject/email_body/...), or None if the DB is empty.
    """
    emails = get_db().get_all_emails()
    if not emails:
        st.info("No emails in the database yet. Run the analyzer first "
                "(`python main.py` → option 2), then refresh.")
        return None

    def _label(e: Dict) -> str:
        subj = (e.get("subject") or "(no subject)")[:50]
        return f"#{e['id']} · {subj} · {e.get('sender', '')[:25]}"

    selected = st.selectbox("Choose an email", emails, format_func=_label, key=key)
    return selected


def render_sources(sources: List[Dict]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            score = s.get("score")
            score_txt = f" · score {score}" if score is not None else ""
            st.markdown(
                f"- **{s.get('subject', '(no subject)')}** — "
                f"{s.get('sender', '')}{score_txt}"
            )


def render_unavailable(reason: Optional[str], what: str = "This feature") -> None:
    st.warning(f"{what} needs the local LLM. {reason or ''}")
    st.markdown(
        "**To enable it:**\n"
        "1. `pip install -r requirements.txt`\n"
        "2. Install [Ollama](https://ollama.com) and run `ollama serve`\n"
        "3. `ollama pull llama3`"
    )
