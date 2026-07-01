"""
Unit Tests — Chat with Emails  (Phase 3)
========================================
Verifies that ChatService grounds answers in retrieved emails, persists turns,
returns sources, and degrades gracefully — using fakes (no daemon / no deps).

Run:
    python -m unittest tests.test_chat -v
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.database import DatabaseManager
from src.services.chat_service import ChatService


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeRag:
    def __init__(self, available=True, hits=None):
        self._available = available
        self._hits = hits or []

    def is_available(self):
        return self._available

    def unavailable_reason(self):
        return None if self._available else "rag offline"

    def search(self, query, top_k=5):
        return self._hits[:top_k]


class FakeChatClient:
    model = "fake-llm"
    reason = None

    def __init__(self, available=True):
        self._available = available
        self.last_messages = None

    def is_available(self, refresh=False):
        return self._available

    def chat(self, messages, json_mode=False):
        self.last_messages = messages
        return "Microsoft asked for a contract renewal quote."

    def generate(self, prompt, system=None, json_mode=False):
        return "summary of conversation"


def _hit(email_id, sender, subject, body, score=0.9):
    return {
        "id": f"email_{email_id}",
        "document": f"{subject}\n\n{body}",
        "metadata": {"email_id": email_id, "sender": sender,
                     "subject": subject, "timestamp": "2025-10-01"},
        "distance": 1 - score,
        "score": score,
    }


class TestChatService(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = DatabaseManager(db_path=self.tmp.name)
        self.hits = [
            _hit(1, "alice@microsoft.com", "Contract renewal",
                 "Please send a renewal quote by Friday."),
            _hit(2, "bob@corp.com", "Lunch", "Grab lunch tomorrow?"),
        ]
        self.client = FakeChatClient()
        self.chat = ChatService(rag=FakeRag(hits=self.hits),
                                client=self.client, db=self.db)

    def tearDown(self):
        self.db = None
        os.unlink(self.tmp.name)

    def test_answer_and_sources(self):
        r = self.chat.ask("What did Microsoft ask?", conversation_id="c1")
        self.assertTrue(r["available"])
        self.assertIn("contract", r["answer"].lower())
        self.assertEqual(len(r["sources"]), 2)
        self.assertEqual(r["sources"][0]["sender"], "alice@microsoft.com")

    def test_context_passed_to_llm(self):
        self.chat.ask("What did Microsoft ask?", conversation_id="c1")
        # The retrieved email subject must appear in the prompt sent to the LLM.
        joined = " ".join(m["content"] for m in self.client.last_messages)
        self.assertIn("Contract renewal", joined)
        self.assertIn("Relevant emails", joined)

    def test_history_persisted(self):
        self.chat.ask("Q1", conversation_id="c1")
        self.chat.ask("Q2", conversation_id="c1")
        history = self.chat.history("c1")
        # 2 user + 2 assistant
        self.assertEqual(len(history), 4)
        self.assertEqual(history[0]["role"], "user")
        self.assertEqual(history[1]["role"], "assistant")

    def test_prior_history_included(self):
        self.chat.ask("First question", conversation_id="c1")
        self.chat.ask("Second question", conversation_id="c1")
        joined = " ".join(m["content"] for m in self.client.last_messages)
        self.assertIn("First question", joined)

    def test_empty_question(self):
        r = self.chat.ask("   ", conversation_id="c1")
        self.assertEqual(r["answer"], "")
        self.assertEqual(self.chat.history("c1"), [])

    def test_rag_unavailable_still_answers(self):
        chat = ChatService(rag=FakeRag(available=False),
                           client=FakeChatClient(), db=self.db)
        r = chat.ask("anything", conversation_id="c2")
        self.assertTrue(r["available"])
        self.assertEqual(r["sources"], [])

    def test_llm_unavailable_graceful(self):
        chat = ChatService(rag=FakeRag(hits=self.hits),
                           client=FakeChatClient(available=False), db=self.db)
        r = chat.ask("anything", conversation_id="c3")
        self.assertFalse(r["available"])
        self.assertEqual(self.db.count_chat_messages(), 0)  # nothing persisted

    def test_conversations_listing(self):
        self.chat.ask("hi", conversation_id="alpha")
        self.chat.ask("hi", conversation_id="beta")
        convos = {c["conversation_id"] for c in self.chat.conversations()}
        self.assertEqual(convos, {"alpha", "beta"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
