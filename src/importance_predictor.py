"""
Email Importance Predictor  (Feature 2)
-----------------------------------------
Computes a composite Importance Score from five independent signals:
    1. Sentiment Weight        — negative → higher urgency
    2. Urgency Keyword Score   — domain-specific trigger words
    3. Sender Importance Score — role-based sender classification
    4. Email Length Score      — longer emails tend to carry more weight
    5. Exclamation Count Score — exclamation marks signal emphasis

Importance levels:
    Score < 3  → LOW
    Score 3–6  → MEDIUM
    Score > 6  → HIGH
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List


# ── Configuration ─────────────────────────────────────────────────────────────

URGENCY_KEYWORDS: List[str] = [
    "urgent", "urgently", "immediately", "asap", "as soon as possible",
    "complaint", "issue", "problem", "critical", "deadline", "overdue",
    "emergency", "escalate", "escalation", "action required", "final notice",
    "reminder", "follow up", "follow-up", "warning", "risk", "failure",
    "broken", "not working", "error", "bug", "outage", "downtime",
]

IMPORTANT_SENDER_KEYWORDS: List[str] = [
    "boss", "ceo", "cto", "cfo", "president", "director", "manager",
    "professor", "prof", "dr.", "doctor", "client", "partner", "investor",
    "board", "executive", "vp", "vice president", "head of",
]

SENTIMENT_SCORE_MAP: Dict[str, float] = {
    "negative": 3.0,   # negative → highest concern
    "neutral":  1.0,
    "positive": 0.0,
}

# Length thresholds (character count)
LENGTH_THRESHOLDS = [
    (2000, 3.0),   # very long
    (1000, 2.0),
    (400,  1.0),
    (0,    0.0),
]

# Exclamation thresholds
EXCLAMATION_THRESHOLDS = [
    (5, 2.0),
    (3, 1.5),
    (1, 1.0),
    (0, 0.0),
]


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class ImportanceResult:
    importance_level: str          # LOW / MEDIUM / HIGH
    total_score: float
    sentiment_score: float
    urgency_score: float
    sender_score: float
    length_score: float
    exclamation_score: float
    urgency_keywords_found: List[str] = field(default_factory=list)
    sender_role_detected: str = ""
    breakdown: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.breakdown = {
            "sentiment_weight":   self.sentiment_score,
            "urgency_keywords":   self.urgency_score,
            "sender_importance":  self.sender_score,
            "email_length":       self.length_score,
            "exclamation_count":  self.exclamation_score,
        }

    def to_dict(self) -> dict:
        return {
            "importance_level":       self.importance_level,
            "total_score":            round(self.total_score, 2),
            "urgency_score":          round(self.urgency_score, 2),
            "sender_role_detected":   self.sender_role_detected,
            "urgency_keywords_found": self.urgency_keywords_found,
            "score_breakdown":        {k: round(v, 2) for k, v in self.breakdown.items()},
        }


# ── ImportancePredictor ───────────────────────────────────────────────────────

class ImportancePredictor:
    """
    Stateless importance scorer.  All methods are pure functions; the class
    is provided for easy dependency injection and future extension.
    """

    # ── Public API ────────────────────────────────────────────────────────

    def calculate_importance(
        self,
        email_text: str,
        sender: str,
        sentiment: str,
    ) -> ImportanceResult:
        """
        Compute the composite importance score.

        Parameters
        ----------
        email_text : str
            The full email body (and optionally subject concatenated).
        sender : str
            The raw From: header value, e.g. ``"John Smith <john@corp.com>"``.
        sentiment : str
            Overall sentiment: 'positive', 'neutral', or 'negative'.

        Returns
        -------
        ImportanceResult
        """
        text_lower = email_text.lower() if isinstance(email_text, str) else ""
        sender_lower = sender.lower() if isinstance(sender, str) else ""

        s_score = self._sentiment_weight(sentiment)
        u_score, keywords_found = self._urgency_keyword_score(text_lower)
        snd_score, role_detected = self._sender_importance_score(sender_lower)
        l_score = self._length_score(text_lower)
        e_score = self._exclamation_score(email_text)

        total = s_score + u_score + snd_score + l_score + e_score
        level = self._classify(total)

        return ImportanceResult(
            importance_level=level,
            total_score=total,
            sentiment_score=s_score,
            urgency_score=u_score,
            sender_score=snd_score,
            length_score=l_score,
            exclamation_score=e_score,
            urgency_keywords_found=keywords_found,
            sender_role_detected=role_detected,
        )

    # ── Component scorers ─────────────────────────────────────────────────

    @staticmethod
    def _sentiment_weight(sentiment: str) -> float:
        return SENTIMENT_SCORE_MAP.get(sentiment.lower(), 0.0)

    @staticmethod
    def _urgency_keyword_score(text_lower: str):
        found = [kw for kw in URGENCY_KEYWORDS if kw in text_lower]
        # Cap at 3.0 to prevent keyword stuffing domination
        score = min(len(found) * 0.5, 3.0)
        return score, found

    @staticmethod
    def _sender_importance_score(sender_lower: str):
        for keyword in IMPORTANT_SENDER_KEYWORDS:
            if keyword in sender_lower:
                return 2.0, keyword
        return 0.0, ""

    @staticmethod
    def _length_score(text_lower: str) -> float:
        length = len(text_lower)
        for threshold, score in LENGTH_THRESHOLDS:
            if length >= threshold:
                return score
        return 0.0

    @staticmethod
    def _exclamation_score(text: str) -> float:
        if not isinstance(text, str):
            return 0.0
        count = text.count("!")
        for threshold, score in EXCLAMATION_THRESHOLDS:
            if count >= threshold:
                return score
        return 0.0

    @staticmethod
    def _classify(total: float) -> str:
        if total > 6:
            return "HIGH"
        if total >= 3:
            return "MEDIUM"
        return "LOW"

    # ── Pretty printer ────────────────────────────────────────────────────

    @staticmethod
    def format_report(result: ImportanceResult) -> str:
        lines = ["\n── Importance Score Analysis ───────────────────────────"]
        lines.append(f"  Importance Level    → {result.importance_level}")
        lines.append(f"  Total Score         → {result.total_score:.2f} / 10")
        lines.append("  Score Breakdown:")
        for k, v in result.breakdown.items():
            lines.append(f"    {k:<22}: {v:.2f}")
        if result.urgency_keywords_found:
            lines.append(f"  Urgency Keywords    → {', '.join(result.urgency_keywords_found)}")
        if result.sender_role_detected:
            lines.append(f"  Sender Role Matched → {result.sender_role_detected}")
        lines.append("─" * 55)
        return "\n".join(lines)
