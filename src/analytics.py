"""
Analytics Dashboard  (Feature 4)
----------------------------------
Generates analysis reports and visualisations from stored email data.

Charts produced (saved to ``reports/``):
  1. sentiment_distribution.png  — Pie chart of positive/neutral/negative
  2. importance_distribution.png — Bar chart of LOW/MEDIUM/HIGH counts
  3. sentiment_trend.png         — Line chart of daily sentiment over time
  4. top_negative_senders.png    — Horizontal bar of most negative senders

Uses matplotlib for offline rendering (no browser required).
"""

import os
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Matplotlib import with graceful fallback (headless servers)
try:
    import matplotlib
    matplotlib.use("Agg")           # non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("[Analytics] matplotlib not installed — charts disabled.")


# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "positive":  "#4CAF50",
    "neutral":   "#2196F3",
    "negative":  "#F44336",
    "HIGH":      "#F44336",
    "MEDIUM":    "#FF9800",
    "LOW":       "#4CAF50",
    "accent":    "#7C4DFF",
    "bg":        "#FAFAFA",
}


# ── AnalyticsDashboard ────────────────────────────────────────────────────────

class AnalyticsDashboard:
    """
    Compute and visualise email analytics from database records.

    Parameters
    ----------
    db : DatabaseManager
        An initialised database manager instance.
    output_dir : str
        Directory where chart PNG files are written.
    """

    def __init__(self, db, output_dir: str = "reports"):
        self.db = db
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ── Text report ───────────────────────────────────────────────────────

    def generate_text_report(self) -> str:
        """Return a formatted text analytics summary."""
        stats = self.db.get_summary_stats()
        sentiment = self.db.get_sentiment_counts()
        importance = self.db.get_importance_counts()
        neg_senders = self.db.get_most_negative_senders(top_n=5)
        urgent = self.db.get_most_urgent_emails(top_n=5)

        total = stats["total_emails"] or 1   # prevent division by zero

        def pct(v): return f"{v / total * 100:.1f}%"

        lines = [
            "",
            "═" * 62,
            "  ADVANCED EMAIL ANALYSIS — ANALYTICS REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "═" * 62,
            "",
            "── OVERVIEW ────────────────────────────────────────────────",
            f"  Total Emails Analysed   : {stats['total_emails']}",
            f"  High Importance Emails  : {stats['high_importance']}  ({pct(stats['high_importance'])})",
            f"  Sentiment Shift Detected: {stats['with_sentiment_shift']}  ({pct(stats['with_sentiment_shift'])})",
            f"  Avg Urgency Score       : {stats['avg_urgency_score']:.2f}",
            "",
            "── SENTIMENT DISTRIBUTION ──────────────────────────────────",
            f"  Positive : {sentiment.get('positive', 0):>4}  ({pct(sentiment.get('positive', 0))})",
            f"  Neutral  : {sentiment.get('neutral',  0):>4}  ({pct(sentiment.get('neutral',  0))})",
            f"  Negative : {sentiment.get('negative', 0):>4}  ({pct(sentiment.get('negative', 0))})",
            "",
            "── IMPORTANCE DISTRIBUTION ─────────────────────────────────",
            f"  HIGH   : {importance.get('HIGH',   0):>4}",
            f"  MEDIUM : {importance.get('MEDIUM', 0):>4}",
            f"  LOW    : {importance.get('LOW',    0):>4}",
        ]

        if neg_senders:
            lines += [
                "",
                "── MOST NEGATIVE SENDERS ───────────────────────────────────",
            ]
            for i, s in enumerate(neg_senders, 1):
                lines.append(f"  {i}. {s['sender']:<40} {s['negative_count']} negative email(s)")

        if urgent:
            lines += [
                "",
                "── MOST URGENT EMAILS ──────────────────────────────────────",
            ]
            for i, e in enumerate(urgent, 1):
                subj = (e.get("subject") or "")[:45]
                lines.append(
                    f"  {i}. [{e['importance_level']:<6}] score={e['urgency_score']:.1f}  {subj}"
                )

        lines += ["", "═" * 62]
        return "\n".join(lines)

    def print_report(self) -> None:
        print(self.generate_text_report())

    def save_text_report(self, filename: str = "analytics_report.txt") -> str:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(self.generate_text_report())
        print(f"[Analytics] Report saved → {path}")
        return path

    # ── Charts ────────────────────────────────────────────────────────────

    def generate_all_charts(self) -> List[str]:
        """Generate all charts and return list of saved file paths."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("[Analytics] matplotlib unavailable — skipping charts.")
            return []

        paths = []
        paths.append(self._chart_sentiment_distribution())
        paths.append(self._chart_importance_distribution())
        paths.append(self._chart_sentiment_trend())
        paths.append(self._chart_top_negative_senders())
        paths.append(self._chart_summary_dashboard())
        return [p for p in paths if p]

    # ── Individual chart methods ──────────────────────────────────────────

    def _chart_sentiment_distribution(self) -> Optional[str]:
        counts = self.db.get_sentiment_counts()
        if not counts:
            return None

        labels = list(counts.keys())
        values = list(counts.values())
        colours = [PALETTE.get(lbl, "#999") for lbl in labels]

        fig, ax = plt.subplots(figsize=(6, 6), facecolor=PALETTE["bg"])
        wedges, texts, autotexts = ax.pie(
            values,
            labels=[l.capitalize() for l in labels],
            colors=colours,
            autopct="%1.1f%%",
            startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 2},
        )
        for at in autotexts:
            at.set_fontsize(11)
            at.set_color("white")
            at.set_fontweight("bold")

        ax.set_title("Email Sentiment Distribution", fontsize=15, pad=20)
        path = self._save(fig, "sentiment_distribution.png")
        plt.close(fig)
        return path

    def _chart_importance_distribution(self) -> Optional[str]:
        counts = self.db.get_importance_counts()
        if not counts:
            return None

        order = ["LOW", "MEDIUM", "HIGH"]
        labels = [l for l in order if l in counts]
        values = [counts[l] for l in labels]
        colours = [PALETTE.get(l, "#999") for l in labels]

        fig, ax = plt.subplots(figsize=(7, 4), facecolor=PALETTE["bg"])
        bars = ax.bar(labels, values, color=colours, edgecolor="white", linewidth=1.5)
        ax.bar_label(bars, padding=4, fontsize=12, fontweight="bold")
        ax.set_title("Email Importance Level Distribution", fontsize=14)
        ax.set_xlabel("Importance Level")
        ax.set_ylabel("Email Count")
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)

        path = self._save(fig, "importance_distribution.png")
        plt.close(fig)
        return path

    def _chart_sentiment_trend(self) -> Optional[str]:
        trend_data = self.db.get_sentiment_trend()
        if not trend_data:
            return None

        # Group by day → sentiment → count
        daily: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in trend_data:
            daily[row["day"]][row["overall_sentiment"]] += row["cnt"]

        if len(daily) < 2:
            return None          # not enough data for a trend line

        days = sorted(daily.keys())
        sentiments = ["positive", "neutral", "negative"]

        fig, ax = plt.subplots(figsize=(10, 4), facecolor=PALETTE["bg"])
        for snt in sentiments:
            values = [daily[d].get(snt, 0) for d in days]
            ax.plot(days, values, marker="o", label=snt.capitalize(),
                    color=PALETTE[snt], linewidth=2, markersize=5)

        ax.set_title("Sentiment Trend Over Time", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("Email Count")
        ax.legend()
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
        plt.xticks(rotation=30, ha="right", fontsize=8)
        plt.tight_layout()

        path = self._save(fig, "sentiment_trend.png")
        plt.close(fig)
        return path

    def _chart_top_negative_senders(self) -> Optional[str]:
        senders = self.db.get_most_negative_senders(top_n=8)
        if not senders:
            return None

        names = [s["sender"][:35] for s in senders]
        counts = [s["negative_count"] for s in senders]

        fig, ax = plt.subplots(figsize=(8, max(3, len(names) * 0.6)),
                               facecolor=PALETTE["bg"])
        bars = ax.barh(names[::-1], counts[::-1],
                       color=PALETTE["negative"], edgecolor="white")
        ax.bar_label(bars, padding=4, fontsize=10)
        ax.set_title("Top Negative Email Senders", fontsize=14)
        ax.set_xlabel("Negative Email Count")
        ax.set_facecolor(PALETTE["bg"])
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()

        path = self._save(fig, "top_negative_senders.png")
        plt.close(fig)
        return path

    def _chart_summary_dashboard(self) -> Optional[str]:
        """
        2×2 grid overview chart: sentiment pie, importance bar,
        urgency top-5, shift count.
        """
        stats = self.db.get_summary_stats()
        sentiment = self.db.get_sentiment_counts()
        importance = self.db.get_importance_counts()

        if not sentiment:
            return None

        fig = plt.figure(figsize=(14, 10), facecolor=PALETTE["bg"])
        fig.suptitle("Email Analysis — Summary Dashboard",
                     fontsize=18, fontweight="bold", y=0.98)
        gs = GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # ── (0,0) Sentiment Pie ───────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        labels = list(sentiment.keys())
        values = list(sentiment.values())
        colours = [PALETTE.get(l, "#999") for l in labels]
        ax1.pie(values, labels=[l.capitalize() for l in labels],
                colors=colours, autopct="%1.0f%%", startangle=140,
                wedgeprops={"edgecolor": "white"})
        ax1.set_title("Sentiment Split")

        # ── (0,1) Importance Bar ──────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        order = ["LOW", "MEDIUM", "HIGH"]
        imp_labels = [l for l in order if l in importance]
        imp_values = [importance[l] for l in imp_labels]
        imp_colours = [PALETTE.get(l, "#999") for l in imp_labels]
        bars = ax2.bar(imp_labels, imp_values, color=imp_colours, edgecolor="white")
        ax2.bar_label(bars, padding=3, fontsize=10)
        ax2.set_title("Importance Levels")
        ax2.spines[["top", "right"]].set_visible(False)

        # ── (1,0) KPI tiles ───────────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis("off")
        kpis = [
            ("Total Emails",         str(stats["total_emails"])),
            ("High Importance",      str(stats["high_importance"])),
            ("Sentiment Shifts",     str(stats["with_sentiment_shift"])),
            ("Avg Urgency Score",    f"{stats['avg_urgency_score']:.2f}"),
        ]
        for i, (label, value) in enumerate(kpis):
            y = 0.85 - i * 0.22
            ax3.text(0.1, y, label,  fontsize=11, color="#555")
            ax3.text(0.65, y, value, fontsize=14, fontweight="bold",
                     color=PALETTE["accent"])
        ax3.set_title("Key Metrics", pad=10)

        # ── (1,1) Top urgent emails ───────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 1])
        urgent = self.db.get_most_urgent_emails(top_n=5)
        if urgent:
            u_labels = [
                f"{e.get('subject', '')[:25]}…" if e.get("subject") else f"Email #{e['id']}"
                for e in urgent
            ]
            u_scores = [e["urgency_score"] for e in urgent]
            bars4 = ax4.barh(u_labels[::-1], u_scores[::-1],
                             color=PALETTE["MEDIUM"], edgecolor="white")
            ax4.bar_label(bars4, fmt="%.1f", padding=3, fontsize=9)
            ax4.set_title("Top 5 Urgent Emails")
            ax4.spines[["top", "right"]].set_visible(False)
        else:
            ax4.text(0.5, 0.5, "No data yet", ha="center", va="center",
                     fontsize=12, color="#aaa")
            ax4.axis("off")

        path = self._save(fig, "summary_dashboard.png")
        plt.close(fig)
        return path

    # ── Helpers ───────────────────────────────────────────────────────────

    def _save(self, fig, filename: str) -> str:
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=130, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Analytics] Chart saved → {path}")
        return path
