"""
LLM Service  (Phase 2)
======================
The intelligence layer on top of the existing ML pipeline. Exposes the eight
LLM features, each:

  1. Summary                 2. Action item extraction
  3. Deadline detection      4. Suggested reply
  5. Email categorization    6. Named entity extraction
  7. Risk detection          8. Explain ML importance prediction

Every feature is **cached** in the ``llm_analysis`` table keyed by a hash of
(feature + model + input text), so the LLM is never called twice for identical
input. When the LLM backend is unavailable, each method returns a structured
``{"available": False, "reason": ...}`` dict instead of raising — the ML
pipeline is entirely unaffected.

Dependencies (client, db) are injected for testability.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Callable, Dict, List, Optional

import config
from src.database import DatabaseManager
from src.llm.ollama_client import OllamaClient, LLMError
from src.llm import prompts

logger = logging.getLogger(__name__)


class LlmService:
    CATEGORIES = prompts.CATEGORIES

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        db: Optional[DatabaseManager] = None,
        cache_enabled: Optional[bool] = None,
    ):
        self.client = client if client is not None else OllamaClient()
        self.db = db if db is not None else DatabaseManager()
        self.cache_enabled = (config.LLM_CACHE_ENABLED
                              if cache_enabled is None else cache_enabled)

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self.client.is_available()

    def unavailable_reason(self) -> Optional[str]:
        return self.client.reason

    # ── Core caching/runner ───────────────────────────────────────────────

    @staticmethod
    def _compose(email: Dict) -> Dict[str, str]:
        return {
            "subject": email.get("subject", "") or "",
            "sender": email.get("from", email.get("sender", "")) or "",
            "body": email.get("body", email.get("email_body", "")) or "",
        }

    def _cache_key(self, feature: str, text: str) -> str:
        raw = f"{feature}|{self.client.model}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _unavailable(self, feature: str) -> Dict:
        return {"available": False, "feature": feature,
                "reason": self.unavailable_reason() or "LLM unavailable"}

    def _run(self, feature: str, cache_text: str, email_id: Optional[int],
             produce: Callable[[], Dict]) -> Dict:
        """
        Cache-aware feature runner.

        ``produce`` is a no-arg callable that performs the actual LLM call and
        returns the (available) result dict. It is only invoked on a cache miss.
        """
        key = self._cache_key(feature, cache_text)

        if self.cache_enabled:
            cached = self.db.get_llm_cache(key)
            if cached:
                result = json.loads(cached["result"])
                result["cached"] = True
                return result

        if not self.is_available():
            return self._unavailable(feature)

        try:
            result = produce()
        except LLMError as exc:
            logger.error("[LLM] %s failed: %s", feature, exc)
            return {"available": False, "feature": feature, "reason": str(exc)}

        result.setdefault("available", True)
        result["cached"] = False
        if self.cache_enabled:
            self.db.save_llm_cache(
                cache_key=key, feature=feature, model=self.client.model,
                result=json.dumps(result), email_id=email_id,
            )
        return result

    def _gen_json(self, system: str, prompt: str) -> Dict:
        """Generate with JSON mode and parse; ``{}`` on parse failure."""
        raw = self.client.generate(prompt, system=system, json_mode=True)
        try:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, TypeError):
            logger.warning("[LLM] Non-JSON response: %r", raw[:120])
            return {}

    # ── 1. Summary ────────────────────────────────────────────────────────

    def summarize(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)

        def produce() -> Dict:
            text = self.client.generate(
                prompts.build_summary(**e), system=prompts.SUMMARY_SYSTEM)
            return {"summary": text}

        return self._run("summary", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 2. Action items ───────────────────────────────────────────────────

    def extract_action_items(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)

        def produce() -> Dict:
            data = self._gen_json(prompts.ACTION_SYSTEM, prompts.build_action_items(**e))
            items = data.get("action_items") or []
            return {"action_items": [str(x) for x in items]}

        return self._run("action_items", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 3. Deadlines ──────────────────────────────────────────────────────

    def detect_deadlines(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)

        def produce() -> Dict:
            data = self._gen_json(prompts.DEADLINE_SYSTEM, prompts.build_deadlines(**e))
            return {"deadlines": data.get("deadlines") or []}

        return self._run("deadlines", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 4. Suggested reply ────────────────────────────────────────────────

    def suggest_reply(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)

        def produce() -> Dict:
            text = self.client.generate(
                prompts.build_reply(**e), system=prompts.REPLY_SYSTEM)
            return {"reply": text}

        return self._run("reply", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 5. Categorization ─────────────────────────────────────────────────

    def categorize(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)

        def produce() -> Dict:
            data = self._gen_json(prompts.CATEGORY_SYSTEM, prompts.build_category(**e))
            category = str(data.get("category", "General")).strip().title()
            if category not in self.CATEGORIES:
                category = "General"
            return {"category": category}

        return self._run("category", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 6. Named entities ─────────────────────────────────────────────────

    def extract_entities(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)
        keys = ("people", "companies", "dates", "products", "locations")

        def produce() -> Dict:
            data = self._gen_json(prompts.ENTITIES_SYSTEM, prompts.build_entities(**e))
            return {"entities": {k: [str(x) for x in (data.get(k) or [])] for k in keys}}

        return self._run("entities", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 7. Risk detection ─────────────────────────────────────────────────

    def detect_risks(self, email: Dict, email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)

        def produce() -> Dict:
            data = self._gen_json(prompts.RISK_SYSTEM, prompts.build_risk(**e))
            level = str(data.get("risk_level", "low")).strip().lower()
            if level not in ("low", "medium", "high"):
                level = "low"
            return {"risk_level": level, "risks": [str(x) for x in (data.get("risks") or [])]}

        return self._run("risk", json.dumps(e, sort_keys=True), email_id, produce)

    # ── 8. Explain ML importance prediction ───────────────────────────────

    def explain_importance(self, email: Dict, importance_level: str,
                           breakdown: Optional[Dict] = None,
                           email_id: Optional[int] = None) -> Dict:
        e = self._compose(email)
        breakdown = breakdown or {}
        # Cache key must include the ML inputs, not just the email text.
        cache_text = json.dumps(
            {**e, "level": importance_level, "breakdown": breakdown}, sort_keys=True)

        def produce() -> Dict:
            text = self.client.generate(
                prompts.build_explain_importance(
                    importance_level=importance_level, breakdown=breakdown, **e),
                system=prompts.EXPLAIN_SYSTEM)
            return {"explanation": text, "importance_level": importance_level}

        return self._run("explain", cache_text, email_id, produce)

    # ── Convenience: run everything ───────────────────────────────────────

    def analyze_all(self, email: Dict, email_id: Optional[int] = None,
                    importance_level: str = "", breakdown: Optional[Dict] = None) -> Dict:
        """Run every feature and return a combined dict (used by the UI)."""
        result = {
            "available": self.is_available(),
            "summary": self.summarize(email, email_id),
            "action_items": self.extract_action_items(email, email_id),
            "deadlines": self.detect_deadlines(email, email_id),
            "reply": self.suggest_reply(email, email_id),
            "category": self.categorize(email, email_id),
            "entities": self.extract_entities(email, email_id),
            "risk": self.detect_risks(email, email_id),
        }
        if importance_level:
            result["explain"] = self.explain_importance(
                email, importance_level, breakdown, email_id)
        return result
