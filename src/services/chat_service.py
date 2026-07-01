"""
Chat Service  (Phase 3 — Chat with Emails)
==========================================
A Retrieval-Augmented chat assistant. For each question it:

  1. Retrieves the most relevant emails via :class:`RagService` (semantic search).
  2. Builds a prompt: system instructions + rolling conversation memory +
     recent turns + the retrieved email context + the question.
  3. Calls the local LLM (:meth:`OllamaClient.chat`).
  4. Persists the turn to ``chat_history`` and refreshes ``conversation_memory``.

Answers like *"What did Microsoft ask last week?"*, *"Which emails mention
Kubernetes?"*, *"What invoices are pending?"* are grounded in the indexed emails.

Graceful degradation
---------------------
* If the LLM is unavailable, :meth:`ask` returns a structured message instead of
  raising.
* If RAG is unavailable, it still answers but flags that no email context could
  be retrieved (rather than failing).

Dependencies (rag, client, db) are injected and share one database.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional

import config
from src.database import DatabaseManager
from src.llm.ollama_client import OllamaClient, LLMError
from src.llm import prompts
from src.services.rag_service import RagService

logger = logging.getLogger(__name__)


class ChatService:
    # Recent turns kept verbatim in the prompt; older turns fold into memory.
    MAX_HISTORY_TURNS = 6
    # Only summarise into long-term memory once a conversation gets long.
    MEMORY_THRESHOLD = 12

    def __init__(
        self,
        rag: Optional[RagService] = None,
        client: Optional[OllamaClient] = None,
        db: Optional[DatabaseManager] = None,
    ):
        self.db = db if db is not None else DatabaseManager()
        self.rag = rag if rag is not None else RagService(db=self.db)
        self.client = client if client is not None else OllamaClient()

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Chat needs at least the LLM. RAG is optional (degrades to no context)."""
        return self.client.is_available()

    def unavailable_reason(self) -> Optional[str]:
        return self.client.reason

    # ── Context formatting ────────────────────────────────────────────────

    @staticmethod
    def _format_context(hits: List[Dict]) -> str:
        lines = []
        for i, hit in enumerate(hits, 1):
            meta = hit.get("metadata", {})
            snippet = (hit.get("document", "") or "").replace("\n", " ")[:300]
            lines.append(
                f"[{i}] From: {meta.get('sender', '')} | "
                f"Subject: {meta.get('subject', '')} | "
                f"Date: {meta.get('timestamp', '')}\n    {snippet}"
            )
        return "\n".join(lines)

    @staticmethod
    def _source(hit: Dict) -> Dict:
        meta = hit.get("metadata", {})
        return {
            "email_id": meta.get("email_id"),
            "sender": meta.get("sender", ""),
            "subject": meta.get("subject", ""),
            "timestamp": meta.get("timestamp", ""),
            "score": hit.get("score"),
        }

    # ── Prompt assembly ───────────────────────────────────────────────────

    def _build_messages(self, conversation_id: str, question: str,
                        context_block: str) -> List[Dict]:
        messages: List[Dict] = [{"role": "system", "content": prompts.CHAT_SYSTEM}]

        memory = self.db.get_conversation_memory(conversation_id)
        if memory:
            messages.append({
                "role": "system",
                "content": f"Conversation so far: {memory['summary']}",
            })

        history = self.db.get_chat_history(conversation_id)
        for msg in history[-self.MAX_HISTORY_TURNS:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        if context_block:
            user_content = (
                f"Relevant emails:\n{context_block}\n\nQuestion: {question}"
            )
        else:
            user_content = (
                "(No relevant emails were retrieved from the index.)\n\n"
                f"Question: {question}"
            )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ── Public API ────────────────────────────────────────────────────────

    def ask(self, question: str, conversation_id: str = "default",
            top_k: Optional[int] = None) -> Dict:
        """
        Answer a question about the indexed emails.

        Returns
        -------
        dict
            ``{available, answer, sources, conversation_id, reason?}``
        """
        question = (question or "").strip()
        if not question:
            return {"available": True, "answer": "", "sources": [],
                    "conversation_id": conversation_id}

        if not self.is_available():
            return {
                "available": False,
                "answer": "The local LLM is unavailable, so I can't chat right now.",
                "reason": self.unavailable_reason(),
                "sources": [],
                "conversation_id": conversation_id,
            }

        # 1. Retrieve context (optional — degrade gracefully if RAG is down).
        sources: List[Dict] = []
        context_block = ""
        if self.rag.is_available():
            hits = self.rag.search(question, top_k=top_k or config.RAG_TOP_K)
            context_block = self._format_context(hits)
            sources = [self._source(h) for h in hits]
        else:
            logger.warning("[Chat] RAG unavailable: %s", self.rag.unavailable_reason())

        # 2. Build prompt and call the LLM.
        messages = self._build_messages(conversation_id, question, context_block)
        try:
            answer = self.client.chat(messages)
        except LLMError as exc:
            return {"available": False, "answer": "", "reason": str(exc),
                    "sources": sources, "conversation_id": conversation_id}

        # 3. Persist the turn.
        self.db.save_chat_message(conversation_id, "user", question)
        self.db.save_chat_message(
            conversation_id, "assistant", answer, sources=json.dumps(sources))

        # 4. Refresh long-term memory if the conversation is getting long.
        self._maybe_update_memory(conversation_id)

        return {"available": True, "answer": answer, "sources": sources,
                "conversation_id": conversation_id}

    # ── Conversation memory ───────────────────────────────────────────────

    def _maybe_update_memory(self, conversation_id: str) -> None:
        history = self.db.get_chat_history(conversation_id)
        if len(history) < self.MEMORY_THRESHOLD:
            return

        older = history[:-self.MAX_HISTORY_TURNS]
        convo_text = "\n".join(f"{m['role']}: {m['content']}" for m in older)
        try:
            summary = self.client.generate(
                f"Summarise this conversation:\n\n{convo_text}",
                system=prompts.MEMORY_SUMMARY_SYSTEM,
            )
            if summary:
                self.db.save_conversation_memory(conversation_id, summary)
        except LLMError as exc:
            logger.warning("[Chat] Memory update skipped: %s", exc)

    # ── History helpers ───────────────────────────────────────────────────

    def history(self, conversation_id: str = "default") -> List[Dict]:
        return self.db.get_chat_history(conversation_id)

    def conversations(self) -> List[Dict]:
        return self.db.list_conversations()
