"""
Ollama Client  (Phase 2 — LLM layer)
====================================
Minimal client for a **local** Ollama daemon, talking to its REST API over
stdlib ``urllib`` — so it needs no third-party package and works as long as the
daemon is running (``ollama serve``).

Local-only: this never contacts OpenAI / Anthropic / Google. The base URL and
model come from :mod:`config` (default ``http://localhost:11434`` / ``llama3``).

Graceful degradation
---------------------
* :meth:`is_available` probes the daemon (``GET /api/tags``) with a short
  timeout and caches the result. If the daemon is down or ``LLM_ENABLED`` is
  false, it returns ``False`` with a human-readable :attr:`reason`.
* :meth:`generate` / :meth:`chat` raise :class:`LLMError` when unavailable;
  callers (the service layer) catch this and return an "unavailable" result.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)

# Short timeout for the availability probe so the UI never hangs on a dead daemon.
_PROBE_TIMEOUT = 3
# Re-probe availability at most this often (seconds), so a recovered daemon or a
# newly-pulled model is picked up without restarting a long-lived process.
_AVAIL_TTL = 10


class LLMError(RuntimeError):
    """Raised when an LLM call is attempted but the backend is unavailable."""


class OllamaClient:
    """
    Parameters
    ----------
    base_url : str
        Ollama daemon URL. Defaults to ``config.OLLAMA_BASE_URL``.
    model : str
        Model name (``llama3`` | ``gemma3`` | ``mistral``). Defaults to
        ``config.LLM_MODEL``.
    timeout : int
        Per-request timeout (seconds) for generation.
    temperature : float
        Sampling temperature.
    """

    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[int] = None, temperature: Optional[float] = None):
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.LLM_MODEL
        self.timeout = timeout or config.LLM_TIMEOUT
        self.temperature = (temperature if temperature is not None
                            else config.LLM_TEMPERATURE)
        self._available: Optional[bool] = None
        self._reason: Optional[str] = None
        self._models: List[str] = []
        self._daemon_up: bool = False
        self._checked_at: float = 0.0

    # ── HTTP helpers ──────────────────────────────────────────────────────

    def _get(self, path: str, timeout: int) -> Dict:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _post(self, path: str, payload: Dict, timeout: int) -> Dict:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self, refresh: bool = False) -> bool:
        # Re-probe on a short TTL rather than caching the result forever. This
        # matters under Streamlit, where the client is cached for the whole
        # session: without a TTL, a daemon that was momentarily down (or a model
        # pulled after startup) would stay "unavailable" until the app restarts.
        now = time.monotonic()
        if (not refresh and self._available is not None
                and (now - self._checked_at) < _AVAIL_TTL):
            return self._available
        self._checked_at = now

        if not config.LLM_ENABLED:
            self._available = False
            self._reason = "LLM disabled via config (LLM_ENABLED=false)"
            return False

        try:
            tags = self._get("/api/tags", timeout=_PROBE_TIMEOUT)
            self._models = [m.get("name", "") for m in tags.get("models", [])]
            self._daemon_up = True
        except Exception as exc:
            self._daemon_up = False
            self._available = False
            self._reason = f"Ollama daemon unreachable at {self.base_url} ({exc})"
            logger.warning("[Ollama] %s", self._reason)
            return False

        # Daemon is reachable — now require the configured model to be pulled.
        # Reporting unavailable here (instead of letting generate() 404 for every
        # feature) gives the user ONE actionable message.
        if not self._has_model():
            installed = ", ".join(self._models) if self._models else "none"
            self._available = False
            self._reason = (
                f"Model '{self.model}' is not installed in Ollama. "
                f"Run: ollama pull {self.model}  (installed: {installed})"
            )
            logger.warning("[Ollama] %s", self._reason)
            return False

        self._available = True
        self._reason = None
        return True

    def daemon_reachable(self) -> bool:
        """True if the Ollama daemon responded, regardless of model availability."""
        self.is_available()
        return getattr(self, "_daemon_up", False)

    def _has_model(self) -> bool:
        base = self.model.split(":")[0]
        return any(m == self.model or m.split(":")[0] == base for m in self._models)

    @property
    def reason(self) -> Optional[str]:
        if self._available is None:
            self.is_available()
        return self._reason

    def list_models(self) -> List[str]:
        self.is_available()
        return list(self._models)

    # ── Generation ────────────────────────────────────────────────────────

    def generate(self, prompt: str, system: Optional[str] = None,
                 json_mode: bool = False) -> str:
        """Single-turn completion. Raises :class:`LLMError` if unavailable."""
        if not self.is_available():
            raise LLMError(self._reason or "LLM unavailable")

        payload: Dict = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if system:
            payload["system"] = system
        if json_mode:
            payload["format"] = "json"

        try:
            resp = self._post("/api/generate", payload, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise LLMError(self._http_error_message(exc, "generation")) from exc
        except Exception as exc:
            raise LLMError(f"generation failed: {exc}") from exc
        return (resp.get("response") or "").strip()

    def _http_error_message(self, exc: "urllib.error.HTTPError", op: str) -> str:
        if exc.code == 404:
            return (f"model '{self.model}' not found — run: ollama pull {self.model}")
        return f"{op} failed: HTTP {exc.code} {exc.reason}"

    def chat(self, messages: List[Dict], json_mode: bool = False) -> str:
        """
        Multi-turn chat (used by the Phase 3 assistant).

        ``messages`` is a list of ``{"role": "system|user|assistant",
        "content": str}`` dicts.
        """
        if not self.is_available():
            raise LLMError(self._reason or "LLM unavailable")

        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            resp = self._post("/api/chat", payload, timeout=self.timeout)
        except urllib.error.HTTPError as exc:
            raise LLMError(self._http_error_message(exc, "chat")) from exc
        except Exception as exc:
            raise LLMError(f"chat failed: {exc}") from exc
        return (resp.get("message", {}).get("content") or "").strip()
