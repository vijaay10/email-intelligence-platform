"""
Cached service factory for the Streamlit UI.
============================================
Streamlit re-runs the whole script on every interaction, so heavy objects
(database connections, embedding models, LLM clients) are cached with
``st.cache_resource`` to build them once per session.

The active LLM model is held in ``st.session_state`` (set on the LLM Settings
page) so the user can switch models without restarting the app.
"""

from __future__ import annotations

import streamlit as st

import config
from src.database import DatabaseManager
from src.services.rag_service import RagService
from src.services.llm_service import LlmService
from src.services.chat_service import ChatService
from src.llm.ollama_client import OllamaClient


@st.cache_resource
def get_db() -> DatabaseManager:
    return DatabaseManager()


@st.cache_resource
def get_rag() -> RagService:
    return RagService(db=get_db())


@st.cache_resource
def get_analyzer():
    """The original ML pipeline orchestrator (lazy — imports scikit-learn)."""
    from src.sentiment_analyzer import EmailSentimentAnalyzer
    return EmailSentimentAnalyzer(model_path=str(config.MODEL_PATH),
                                  db_path=str(config.DB_PATH))


@st.cache_resource
def get_llm_client(model: str) -> OllamaClient:
    """Cached per model name, so switching models builds a fresh client."""
    return OllamaClient(model=model)


def active_model() -> str:
    """The user-selected model, defaulting to config."""
    return st.session_state.get("llm_model", config.LLM_MODEL)


def get_llm_service() -> LlmService:
    return LlmService(client=get_llm_client(active_model()), db=get_db())


def get_chat_service() -> ChatService:
    return ChatService(rag=get_rag(),
                       client=get_llm_client(active_model()),
                       db=get_db())
