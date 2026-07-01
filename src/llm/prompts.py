"""
Prompt Templates  (Phase 2 — LLM layer)
=======================================
Centralised system + user prompts for every LLM feature. Keeping them here
(rather than inline in the service) makes them easy to tune and to review.

Each ``build_*`` function returns the user prompt string; the paired ``*_SYSTEM``
constant is the system message. JSON-returning features instruct the model to
emit a specific schema and are called with ``json_mode=True``.
"""

from __future__ import annotations

# Categories the platform recognises (Email Categorization feature).
CATEGORIES = [
    "Support", "Finance", "HR", "Meeting", "Invoice",
    "Marketing", "Legal", "Security", "Spam", "General",
]


def _email_block(subject: str, sender: str, body: str) -> str:
    return (
        f"From: {sender or 'unknown'}\n"
        f"Subject: {subject or '(no subject)'}\n\n"
        f"{body or ''}"
    ).strip()


# ── Chat with Emails (Phase 3) ─────────────────────────────────────────────────
CHAT_SYSTEM = (
    "You are an assistant that answers questions about the user's emails. "
    "Answer using ONLY the relevant emails provided to you as context. "
    "When you use an email, cite it by sender and subject. "
    "If the provided emails do not contain the answer, say you couldn't find it "
    "in the emails — do not invent details. Be concise and direct."
)

MEMORY_SUMMARY_SYSTEM = (
    "You summarise an ongoing chat conversation factually in 3-4 sentences, "
    "preserving any facts, names, and decisions that future questions may rely on."
)


# ── 1. Summary ─────────────────────────────────────────────────────────────────
SUMMARY_SYSTEM = (
    "You are an assistant that writes concise, factual summaries of emails. "
    "Respond with 1-2 plain sentences. No preamble, no bullet points."
)


def build_summary(subject: str, sender: str, body: str) -> str:
    return f"Summarise this email in 1-2 sentences:\n\n{_email_block(subject, sender, body)}"


# ── 2. Action items ────────────────────────────────────────────────────────────
ACTION_SYSTEM = (
    "You extract concrete action items / tasks from emails. "
    'Respond ONLY as JSON: {"action_items": ["...", "..."]}. '
    "If there are none, return an empty list."
)


def build_action_items(subject: str, sender: str, body: str) -> str:
    return f"Extract the action items from this email:\n\n{_email_block(subject, sender, body)}"


# ── 3. Deadlines ───────────────────────────────────────────────────────────────
DEADLINE_SYSTEM = (
    "You extract deadlines and time references from emails (e.g. 'today', "
    "'tomorrow', '5 PM', '31 July'). "
    'Respond ONLY as JSON: {"deadlines": [{"text": "...", "when": "..."}]}. '
    "'text' is the phrase as written; 'when' is your normalised interpretation. "
    "Empty list if none."
)


def build_deadlines(subject: str, sender: str, body: str) -> str:
    return f"Extract deadlines / dates from this email:\n\n{_email_block(subject, sender, body)}"


# ── 4. Suggested reply ─────────────────────────────────────────────────────────
REPLY_SYSTEM = (
    "You draft professional, concise email replies. Match a polite business "
    "tone. Return only the reply body — no subject line, no placeholders like "
    "[Name] unless necessary."
)


def build_reply(subject: str, sender: str, body: str) -> str:
    return f"Draft a professional reply to this email:\n\n{_email_block(subject, sender, body)}"


# ── 5. Categorization ──────────────────────────────────────────────────────────
CATEGORY_SYSTEM = (
    "You classify an email into exactly one category from this fixed list: "
    + ", ".join(CATEGORIES) + ". "
    'Respond ONLY as JSON: {"category": "<one category>"}.'
)


def build_category(subject: str, sender: str, body: str) -> str:
    return f"Classify this email:\n\n{_email_block(subject, sender, body)}"


# ── 6. Named entities ──────────────────────────────────────────────────────────
ENTITIES_SYSTEM = (
    "You extract named entities from emails. "
    'Respond ONLY as JSON with these keys: '
    '{"people": [], "companies": [], "dates": [], "products": [], "locations": []}. '
    "Use empty lists where nothing applies."
)


def build_entities(subject: str, sender: str, body: str) -> str:
    return f"Extract named entities from this email:\n\n{_email_block(subject, sender, body)}"


# ── 7. Risk detection ──────────────────────────────────────────────────────────
RISK_SYSTEM = (
    "You assess business risk in an email. Consider: high priority, client "
    "escalation, security issue, legal issue, financial risk. "
    'Respond ONLY as JSON: {"risk_level": "low|medium|high", "risks": ["..."]}. '
    "'risks' lists the specific risk flags found (empty if none)."
)


def build_risk(subject: str, sender: str, body: str) -> str:
    return f"Assess the risks in this email:\n\n{_email_block(subject, sender, body)}"


# ── 8. Explain ML importance prediction ────────────────────────────────────────
EXPLAIN_SYSTEM = (
    "You explain, in 1-2 sentences, WHY a rule-based system rated an email's "
    "importance the way it did. Be specific and reference the email's content. "
    "No preamble."
)


def build_explain_importance(subject: str, sender: str, body: str,
                             importance_level: str, breakdown: dict) -> str:
    factors = ", ".join(f"{k}={v}" for k, v in (breakdown or {}).items())
    return (
        f"The importance model rated this email '{importance_level}'.\n"
        f"Score factors: {factors}\n\n"
        f"Explain why this rating makes sense given the email:\n\n"
        f"{_email_block(subject, sender, body)}"
    )
