"""
Email Sentiment Analyser  (Orchestrator)
-----------------------------------------
Ties together every module in the pipeline:

  EmailFetcher → Preprocessor → SentenceAnalyzer →
  ImportancePredictor → DatabaseManager → AnalyticsDashboard

Usage
-----
    analyser = EmailSentimentAnalyzer()
    result   = analyser.analyse_email(email_dict)
    analyser.print_result(result)
"""

import os
import logging
from typing import Dict, List, Optional

from src.nlp_model import SentimentModel
from src.sentence_analyzer import SentenceAnalyzer
from src.importance_predictor import ImportancePredictor
from src.database import DatabaseManager
from src.analytics import AnalyticsDashboard

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "sentiment_model.pkl"
)


class EmailSentimentAnalyzer:
    """
    High-level analyser.  Loads the model once; each call to
    :meth:`analyse_email` runs the full pipeline and persists results.

    Parameters
    ----------
    model_path : str
        Path to the trained ``sentiment_model.pkl``.
    db_path : str | None
        SQLite database path.  Defaults to ``db/email_analysis.db``.
    reports_dir : str
        Directory for analytics charts and reports.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        db_path: Optional[str] = None,
        reports_dir: str = "reports",
    ):
        # ── Load model ────────────────────────────────────────────────────
        self.model = SentimentModel()
        if os.path.exists(model_path):
            self.model.load_model(model_path)
        else:
            logger.warning(
                "Model not found at %s — run scripts/train_model.py first.", model_path
            )

        # ── Initialise sub-components ─────────────────────────────────────
        self.sentence_analyser = SentenceAnalyzer(self.model)
        self.importance_predictor = ImportancePredictor()
        self.db = DatabaseManager(db_path)
        self.dashboard = AnalyticsDashboard(self.db, output_dir=reports_dir)

    # ── Single email ──────────────────────────────────────────────────────

    def analyse_email(self, email_data: Dict, persist: bool = True) -> Dict:
        """
        Full pipeline for one email.

        Parameters
        ----------
        email_data : dict
            Keys: id, subject, from, date, body
        persist : bool
            Whether to save results to the database.

        Returns
        -------
        dict  —  complete analysis result
        """
        subject = email_data.get("subject", "") or ""
        body    = email_data.get("body",    "") or ""
        sender  = email_data.get("from",    "") or ""
        date    = email_data.get("date",    "") or ""

        # ── Step 1: sentence-level sentiment ─────────────────────────────
        sent_analysis = self.sentence_analyser.analyse(body, subject)

        # ── Step 2: importance scoring ────────────────────────────────────
        full_text = f"{subject}\n{body}"
        importance_result = self.importance_predictor.calculate_importance(
            email_text=full_text,
            sender=sender,
            sentiment=sent_analysis["overall_sentiment"],
        )

        # ── Step 3: assemble result ───────────────────────────────────────
        result = {
            "email_id":          email_data.get("id", "unknown"),
            "subject":           subject,
            "sender":            sender,
            "date":              date,
            # Sentence analysis
            "sentences":         sent_analysis["sentences"],
            "sentence_count":    sent_analysis["sentence_count"],
            "sentiment_shift":   sent_analysis["sentiment_shift"],
            "shift_description": sent_analysis["shift_description"],
            "overall_sentiment": sent_analysis["overall_sentiment"],
            # Importance
            "importance_level":  importance_result.importance_level,
            "urgency_score":     importance_result.urgency_score,
            "importance_detail": importance_result.to_dict(),
        }

        # ── Step 4: persist to DB ─────────────────────────────────────────
        if persist and self.model.is_trained:
            try:
                db_id = self.db.save_email_analysis(
                    sender=sender,
                    subject=subject,
                    email_body=body,
                    overall_sentiment=result["overall_sentiment"],
                    importance_level=result["importance_level"],
                    sentiment_shift=result["sentiment_shift"],
                    urgency_score=result["urgency_score"],
                    date=date,
                    sentences=sent_analysis["sentences"],
                )
                result["db_id"] = db_id
            except Exception as exc:
                logger.error("DB persist failed: %s", exc)
                result["db_id"] = None

        return result

    # ── Batch ─────────────────────────────────────────────────────────────

    def analyse_emails(self, emails: List[Dict], persist: bool = True) -> List[Dict]:
        """Analyse a list of email dicts and return a list of results."""
        results = []
        total = len(emails)
        print(f"\n[Analyser] Processing {total} email(s)…\n")

        for i, email_data in enumerate(emails, 1):
            subject_preview = (email_data.get("subject") or "")[:50]
            print(f"  [{i}/{total}] {subject_preview}…")
            result = self.analyse_email(email_data, persist=persist)
            results.append(result)

        return results

    # ── Report printing ───────────────────────────────────────────────────

    def print_result(self, result: Dict) -> None:
        """Pretty-print a single analysis result to stdout."""
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  EMAIL ANALYSIS RESULT")
        print(f"{sep}")
        print(f"  From    : {result['sender']}")
        print(f"  Subject : {result['subject']}")
        print(f"  Date    : {result['date']}")

        # Sentence breakdown
        print(f"\n  Sentences ({result['sentence_count']} total):")
        for s in result["sentences"]:
            label = s["sentiment"].upper()
            print(f"    Sentence {s['index']:>2}: [{label:<8}]  {s['sentence'][:75]}")

        # Shift
        shift_tag = "⚠  YES" if result["sentiment_shift"] else "No"
        print(f"\n  Sentiment Shift : {shift_tag}")
        print(f"  Shift Note      : {result['shift_description']}")
        print(f"\n  Overall Sentiment : {result['overall_sentiment'].upper()}")
        print(f"  Importance Level  : {result['importance_level']}")
        print(f"  Urgency Score     : {result['urgency_score']:.2f}")

        # Importance breakdown
        bd = result["importance_detail"].get("score_breakdown", {})
        if bd:
            print("\n  Score Breakdown:")
            for k, v in bd.items():
                print(f"    {k:<24}: {v:.2f}")

        kw = result["importance_detail"].get("urgency_keywords_found", [])
        if kw:
            print(f"\n  Urgency Keywords : {', '.join(kw)}")

        if result.get("db_id"):
            print(f"\n  DB Record ID     : {result['db_id']}")

        print(sep)

    def print_batch_report(self, results: List[Dict]) -> None:
        """Concise summary for a batch of emails."""
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  BATCH ANALYSIS SUMMARY  ({len(results)} emails)")
        print(f"{sep}")

        for i, r in enumerate(results, 1):
            shift = "⚠ SHIFT" if r["sentiment_shift"] else "      "
            print(
                f"  {i:>3}. [{r['overall_sentiment'].upper():<8}] "
                f"[{r['importance_level']:<6}] {shift}  "
                f"{(r['subject'] or '')[:45]}"
            )

        total = len(results)
        sentiments = {}
        importance = {}
        for r in results:
            sentiments[r["overall_sentiment"]] = sentiments.get(r["overall_sentiment"], 0) + 1
            importance[r["importance_level"]]  = importance.get(r["importance_level"],  0) + 1

        print(f"\n  Sentiment   — " + "  ".join(f"{k}: {v}" for k, v in sentiments.items()))
        print(f"  Importance  — " + "  ".join(f"{k}: {v}" for k, v in importance.items()))
        shifts = sum(1 for r in results if r["sentiment_shift"])
        print(f"  Shift Detected in {shifts}/{total} email(s)")
        print(sep)

    # ── Dashboard helper ──────────────────────────────────────────────────

    def run_analytics(self, save_charts: bool = True) -> None:
        """Print text report and optionally generate charts."""
        self.dashboard.print_report()
        if save_charts:
            paths = self.dashboard.generate_all_charts()
            if paths:
                print(f"\n[Analytics] {len(paths)} chart(s) saved.")
        self.dashboard.save_text_report()
