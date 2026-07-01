"""
Unit Tests — LLM layer  (Phase 2)
=================================
Exercises LlmService feature parsing, validation, and caching using a fake
Ollama client (no daemon needed), plus the graceful-degradation path of the
real OllamaClient when no daemon is reachable.

Run:
    python -m unittest tests.test_llm -v
"""

import os
import sys
import json
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.database import DatabaseManager
from src.services.llm_service import LlmService


# ── Fake client ───────────────────────────────────────────────────────────────

class FakeClient:
    """
    Canned Ollama stand-in. Routes by keywords in the system prompt so each
    feature gets a plausible response. Counts calls to verify caching.
    """

    model = "fake-llm"
    reason = None

    def __init__(self):
        self.calls = 0

    def is_available(self, refresh=False) -> bool:
        return True

    def generate(self, prompt, system=None, json_mode=False) -> str:
        self.calls += 1
        sys_l = (system or "").lower()
        if "summaris" in sys_l or "summar" in sys_l:
            return "Production server outage affecting a client."
        if "action item" in sys_l:
            return json.dumps({"action_items": ["Restore server", "Notify client"]})
        if "deadline" in sys_l:
            return json.dumps({"deadlines": [{"text": "5 PM", "when": "today 17:00"}]})
        if "repl" in sys_l:
            return "Thank you for flagging this. We are investigating now."
        if "categor" in sys_l or "classify" in sys_l:
            return json.dumps({"category": "Support"})
        if "named entit" in sys_l or "entit" in sys_l:
            return json.dumps({"people": ["Sarah"], "companies": ["Acme"],
                               "dates": [], "products": [], "locations": ["NYC"]})
        if "risk" in sys_l:
            return json.dumps({"risk_level": "high", "risks": ["client escalation"]})
        if "explain" in sys_l or "why" in sys_l:
            return "Rated HIGH because of urgent keywords and a negative tone."
        return "ok"


class TestLlmService(unittest.TestCase):

    EMAIL = {
        "subject": "URGENT: Server Outage",
        "from": "manager@corp.com",
        "body": "Critical production outage. Please restore by 5 PM and notify the client.",
    }

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = DatabaseManager(db_path=self.tmp.name)
        self.client = FakeClient()
        self.svc = LlmService(client=self.client, db=self.db)

    def tearDown(self):
        self.db = None
        os.unlink(self.tmp.name)

    def test_summary(self):
        r = self.svc.summarize(self.EMAIL, email_id=1)
        self.assertTrue(r["available"])
        self.assertIn("outage", r["summary"].lower())

    def test_action_items(self):
        r = self.svc.extract_action_items(self.EMAIL, email_id=1)
        self.assertEqual(r["action_items"], ["Restore server", "Notify client"])

    def test_deadlines(self):
        r = self.svc.detect_deadlines(self.EMAIL, email_id=1)
        self.assertEqual(r["deadlines"][0]["text"], "5 PM")

    def test_reply(self):
        r = self.svc.suggest_reply(self.EMAIL, email_id=1)
        self.assertIn("investigating", r["reply"].lower())

    def test_category_valid(self):
        r = self.svc.categorize(self.EMAIL, email_id=1)
        self.assertEqual(r["category"], "Support")
        self.assertIn(r["category"], self.svc.CATEGORIES)

    def test_category_falls_back_to_general(self):
        # Force an out-of-list category from the model.
        self.client.generate = lambda *a, **k: json.dumps({"category": "Nonsense"})
        r = self.svc.categorize(self.EMAIL, email_id=2)
        self.assertEqual(r["category"], "General")

    def test_entities(self):
        r = self.svc.extract_entities(self.EMAIL, email_id=1)
        self.assertIn("Sarah", r["entities"]["people"])
        self.assertEqual(r["entities"]["dates"], [])

    def test_risk(self):
        r = self.svc.detect_risks(self.EMAIL, email_id=1)
        self.assertEqual(r["risk_level"], "high")
        self.assertIn("client escalation", r["risks"])

    def test_explain_importance(self):
        r = self.svc.explain_importance(
            self.EMAIL, importance_level="HIGH",
            breakdown={"urgency_keywords": 2.0}, email_id=1)
        self.assertTrue(r["available"])
        self.assertEqual(r["importance_level"], "HIGH")

    def test_caching_avoids_second_call(self):
        self.svc.summarize(self.EMAIL, email_id=1)
        calls_after_first = self.client.calls
        # Second identical call must hit the cache, not the model.
        r2 = self.svc.summarize(self.EMAIL, email_id=1)
        self.assertEqual(self.client.calls, calls_after_first)
        self.assertTrue(r2["cached"])
        self.assertEqual(self.db.count_llm_analyses(), 1)

    def test_bad_json_degrades_gracefully(self):
        self.client.generate = lambda *a, **k: "not json at all"
        r = self.svc.extract_action_items(self.EMAIL, email_id=3)
        self.assertEqual(r["action_items"], [])  # empty, no crash

    def test_analyze_all_shape(self):
        r = self.svc.analyze_all(self.EMAIL, email_id=1,
                                 importance_level="HIGH", breakdown={})
        for key in ("summary", "action_items", "deadlines", "reply",
                    "category", "entities", "risk", "explain"):
            self.assertIn(key, r)


# ── Graceful degradation (no daemon) ───────────────────────────────────────────

class TestLlmGracefulDegradation(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = DatabaseManager(db_path=self.tmp.name)

    def tearDown(self):
        self.db = None
        os.unlink(self.tmp.name)

    def test_no_daemon_returns_unavailable(self):
        from src.llm.ollama_client import OllamaClient
        # Point at a port nothing is listening on.
        client = OllamaClient(base_url="http://127.0.0.1:1", model="llama3")
        if client.is_available():
            self.skipTest("Something IS listening on the probe port")
        svc = LlmService(client=client, db=self.db, cache_enabled=False)
        r = svc.summarize({"subject": "x", "from": "a@b.com", "body": "hi"})
        self.assertFalse(r["available"])
        self.assertIn("reason", r)


if __name__ == "__main__":
    unittest.main(verbosity=2)
