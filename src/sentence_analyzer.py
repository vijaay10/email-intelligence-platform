"""
Sentence-Level Sentiment Analyser  (Feature 1)
------------------------------------------------
Splits an email body into sentences, classifies each one independently,
detects sentiment shifts, and computes the overall email sentiment via
majority vote (with negative-bias tie-breaking for better recall on
high-importance emails).
"""

import re
from collections import Counter
from typing import List, Dict, Tuple

# Optional NLTK — graceful fallback to regex tokeniser if not installed
_NLTK_AVAILABLE = False
try:
    import nltk
    for _res in ["tokenizers/punkt", "tokenizers/punkt_tab"]:
        try:
            nltk.data.find(_res)
        except LookupError:
            nltk.download(_res.split("/")[-1], quiet=True)
    from nltk.tokenize import sent_tokenize as _nltk_sent_tokenize
    _NLTK_AVAILABLE = True
except Exception:
    pass


# ── Helpers ──────────────────────────────────────────────────────────────────

# Regex-based sentence splitter (used when NLTK is unavailable)
_SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def _clean_sentence(s: str) -> str:
    """Strip whitespace; discard very short or punctuation-only fragments."""
    s = s.strip()
    return s if len(s) > 8 else ""


def split_into_sentences(text: str) -> List[str]:
    """
    Tokenise *text* into sentences.
    Uses NLTK punkt if available, otherwise a robust regex fallback.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    try:
        if _NLTK_AVAILABLE:
            raw = _nltk_sent_tokenize(text)
        else:
            # Regex: split on period/!/? followed by whitespace + capital letter
            raw = _SENT_RE.split(text)
    except Exception:
        raw = _SENT_RE.split(text)

    sentences = [_clean_sentence(s) for s in raw]
    return [s for s in sentences if s]


# ── SentenceAnalyzer ─────────────────────────────────────────────────────────

class SentenceAnalyzer:
    """
    Uses a trained :class:`~src.nlp_model.SentimentModel` to perform
    sentence-granularity analysis on email bodies.

    Parameters
    ----------
    model : SentimentModel
        An already-loaded (trained) sentiment model instance.
    """

    # Weights for overall score computation (negative bias for importance)
    SENTIMENT_WEIGHTS = {"positive": 1, "neutral": 0, "negative": -1}

    def __init__(self, model):
        self.model = model

    # ── Core analysis ─────────────────────────────────────────────────────

    def analyse(self, email_body: str, subject: str = "") -> Dict:
        """
        Full sentence-level analysis of an email.

        Returns
        -------
        dict with keys:
          - sentences      : list of {sentence, sentiment, confidence}
          - sentiment_shift: bool
          - overall_sentiment: str
          - shift_description: str
          - sentence_count : int
        """
        text = f"{subject}. {email_body}" if subject else email_body
        sentences = split_into_sentences(text)

        if not sentences:
            return self._empty_result()

        # Classify all sentences in one batch call (efficient)
        predictions = self.model.predict_batch(sentences)

        sentence_results: List[Dict] = []
        for idx, (sent, pred) in enumerate(zip(sentences, predictions), start=1):
            sentence_results.append({
                "index": idx,
                "sentence": sent,
                "sentiment": pred["sentiment"],
                "confidence": round(pred["confidence"], 4),
                "probabilities": pred.get("probabilities", {}),
            })

        shift, shift_desc = self._detect_shift(sentence_results)
        overall = self._compute_overall(sentence_results)

        return {
            "sentences": sentence_results,
            "sentiment_shift": shift,
            "shift_description": shift_desc,
            "overall_sentiment": overall,
            "sentence_count": len(sentence_results),
        }

    # ── Shift detection ───────────────────────────────────────────────────

    @staticmethod
    def _detect_shift(sentence_results: List[Dict]) -> Tuple[bool, str]:
        """
        A shift is detected when the email contains **both** positive and
        negative sentences, indicating mixed emotional tone.
        """
        sentiments = {r["sentiment"] for r in sentence_results}
        has_pos = "positive" in sentiments
        has_neg = "negative" in sentiments

        if has_pos and has_neg:
            # Count how many times polarity flipped sequentially
            labels = [r["sentiment"] for r in sentence_results]
            flips = sum(
                1 for i in range(1, len(labels))
                if labels[i] != labels[i - 1]
                and {labels[i], labels[i - 1]} != {"neutral"}
            )
            desc = (
                f"Mixed tone detected — positive and negative sentences present "
                f"({flips} polarity flip(s))."
            )
            return True, desc

        if len(sentiments) > 1:
            return False, "Sentiment is consistent with minor neutral variations."

        only = next(iter(sentiments))
        return False, f"Uniform {only} tone throughout the email."

    # ── Overall sentiment ─────────────────────────────────────────────────

    def _compute_overall(self, sentence_results: List[Dict]) -> str:
        """
        Majority vote; ties broken toward *negative* (higher importance recall).
        """
        counts = Counter(r["sentiment"] for r in sentence_results)

        # Pure majority
        most_common = counts.most_common()
        if most_common[0][1] > most_common[1][1] if len(most_common) > 1 else True:
            return most_common[0][0]

        # Tie — prefer negative
        tied = {lbl for lbl, cnt in most_common if cnt == most_common[0][1]}
        if "negative" in tied:
            return "negative"
        if "positive" in tied:
            return "positive"
        return "neutral"

    # ── Fallback ──────────────────────────────────────────────────────────

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "sentences": [],
            "sentiment_shift": False,
            "shift_description": "No content to analyse.",
            "overall_sentiment": "neutral",
            "sentence_count": 0,
        }

    # ── Pretty printer ────────────────────────────────────────────────────

    @staticmethod
    def format_report(analysis: Dict) -> str:
        lines = ["\n── Sentence-Level Sentiment Analysis ──────────────────"]
        for s in analysis["sentences"]:
            label = s["sentiment"].upper()
            conf = f"{s['confidence']:.0%}"
            lines.append(f"  Sentence {s['index']:>2}: [{label:<8}] ({conf}) {s['sentence'][:80]}")
        lines.append("")
        lines.append(f"  Sentiment Shift Detected → {'YES ⚠' if analysis['sentiment_shift'] else 'No'}")
        lines.append(f"  Shift Note             → {analysis['shift_description']}")
        lines.append(f"  Overall Email Sentiment → {analysis['overall_sentiment'].upper()}")
        lines.append("─" * 55)
        return "\n".join(lines)
