"""
Advanced Intelligent Email Analysis System
==========================================
Entry point.

Modes
-----
  1. Analyse live Gmail inbox
  2. Analyse demo emails (no Gmail needed)
  3. Retrain model
  4. Run analytics dashboard
  5. Real-time inbox monitoring
"""

import os
import sys
import logging

# ── path bootstrap ────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(name)s  %(message)s",
)

from src.sentiment_analyzer import EmailSentimentAnalyzer
from src.email_fetcher import EmailFetcher

MODEL_PATH = os.path.join(ROOT, "models", "sentiment_model.pkl")


# ── Demo data ─────────────────────────────────────────────────────────────────
DEMO_EMAILS = [
    {
        "id": "demo_001",
        "subject": "URGENT: Server Outage — Immediate Action Required!",
        "from": "manager@corp.com",
        "date": "2025-10-01",
        "body": (
            "We have a critical server outage affecting all production services. "
            "This is an emergency situation and must be resolved asap. "
            "Our clients are complaining and we risk losing major contracts. "
            "However, the backup system is responding well and we have isolated the problem. "
            "Please escalate immediately to the on-call engineer."
        ),
    },
    {
        "id": "demo_002",
        "subject": "Great work on the Q3 project!",
        "from": "sarah@example.com",
        "date": "2025-10-02",
        "body": (
            "I just wanted to say what a fantastic job the whole team did this quarter. "
            "The results exceeded expectations and the client was absolutely thrilled. "
            "The presentation was polished and professional. "
            "Looking forward to another great quarter together!"
        ),
    },
    {
        "id": "demo_003",
        "subject": "Invoice #4521 — Payment Overdue",
        "from": "billing@vendor.com",
        "date": "2025-10-03",
        "body": (
            "This is a reminder that invoice #4521 for $3,200 is now 30 days overdue. "
            "Please arrange payment immediately to avoid service suspension. "
            "We value our relationship and hope to resolve this quickly. "
            "If there is a problem with the invoice, please contact us today."
        ),
    },
    {
        "id": "demo_004",
        "subject": "Meeting notes — Tuesday standup",
        "from": "team@internal.com",
        "date": "2025-10-04",
        "body": (
            "Please find attached the notes from Tuesday's standup. "
            "We covered sprint progress, blockers, and upcoming deadlines. "
            "The team is on track and there are no major issues to report. "
            "Next meeting is Thursday at 10:00 AM."
        ),
    },
    {
        "id": "demo_005",
        "subject": "Complaint: Product defect — refund requested!!",
        "from": "angry.customer@mail.com",
        "date": "2025-10-05",
        "body": (
            "I am extremely disappointed with the product I received. "
            "It arrived broken and the packaging was damaged. "
            "This is completely unacceptable and I demand an immediate refund!! "
            "Your customer service has been unresponsive to my previous emails. "
            "I will escalate this to consumer protection if not resolved today!"
        ),
    },
]


# ── Menu helpers ──────────────────────────────────────────────────────────────

def banner():
    print("\n" + "=" * 62)
    print("  ADVANCED INTELLIGENT EMAIL ANALYSIS SYSTEM")
    print("=" * 62)


def menu() -> str:
    print("\n  Select mode:")
    print("  [1] Analyse Gmail inbox")
    print("  [2] Analyse demo emails (no Gmail needed)")
    print("  [3] Retrain model")
    print("  [4] Run analytics dashboard")
    print("  [5] Monitor inbox in real-time")
    print("  [Q] Quit")
    return input("\n  Choice: ").strip().upper()


def ensure_model(analyser: EmailSentimentAnalyzer) -> bool:
    if not analyser.model.is_trained:
        print("\n[!] Model not found. Training now…")
        from scripts.train_model import train_and_evaluate
        train_and_evaluate()
        analyser.model.load_model(MODEL_PATH)
    return analyser.model.is_trained


# ── Mode handlers ─────────────────────────────────────────────────────────────

def mode_gmail(analyser: EmailSentimentAnalyzer):
    if not ensure_model(analyser):
        print("[!] Training failed. Cannot continue.")
        return

    print("\n  Gmail App Password setup:")
    print("  Google Account > Security > 2-Step Verification > App Passwords")
    email_addr = input("\n  Gmail address : ").strip()
    password   = input("  App Password  : ").strip()
    limit      = int(input("  Emails to fetch (default 10): ").strip() or "10")

    fetcher = EmailFetcher(email_addr, password)
    if not fetcher.connect():
        return

    emails = fetcher.get_inbox_emails(limit=limit)
    fetcher.disconnect()

    if not emails:
        print("  No emails found.")
        return

    results = analyser.analyse_emails(emails)
    analyser.print_batch_report(results)

    show_detail = input("\n  Show full detail for which email? (number or Enter to skip): ").strip()
    if show_detail.isdigit():
        idx = int(show_detail) - 1
        if 0 <= idx < len(results):
            analyser.print_result(results[idx])


def mode_demo(analyser: EmailSentimentAnalyzer):
    if not ensure_model(analyser):
        print("[!] Training failed.")
        return

    print(f"\n  Running analysis on {len(DEMO_EMAILS)} demo emails…")
    results = analyser.analyse_emails(DEMO_EMAILS)
    analyser.print_batch_report(results)

    for r in results:
        analyser.print_result(r)


def mode_retrain():
    from scripts.train_model import train_and_evaluate
    print("\n  Retraining model…")
    data_path = input(f"  CSV path (Enter for default): ").strip() or None
    train_and_evaluate(data_path=data_path or os.path.join(ROOT, "data", "train_data.csv"))


def mode_analytics(analyser: EmailSentimentAnalyzer):
    analyser.run_analytics(save_charts=True)


def mode_monitor(analyser: EmailSentimentAnalyzer):
    if not ensure_model(analyser):
        return

    email_addr = input("  Gmail address : ").strip()
    password   = input("  App Password  : ").strip()
    interval   = int(input("  Poll interval in seconds (default 60): ").strip() or "60")

    fetcher = EmailFetcher(email_addr, password)
    if not fetcher.connect():
        return

    def on_new_email(email_data):
        result = analyser.analyse_email(email_data)
        analyser.print_result(result)

    try:
        fetcher.monitor_inbox(callback=on_new_email, interval=interval)
    finally:
        fetcher.disconnect()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    banner()
    analyser = EmailSentimentAnalyzer(model_path=MODEL_PATH)

    while True:
        choice = menu()
        if choice == "1":
            mode_gmail(analyser)
        elif choice == "2":
            mode_demo(analyser)
        elif choice == "3":
            mode_retrain()
        elif choice == "4":
            mode_analytics(analyser)
        elif choice == "5":
            mode_monitor(analyser)
        elif choice == "Q":
            print("\n  Goodbye!\n")
            break
        else:
            print("  Invalid choice — please try again.")


if __name__ == "__main__":
    main()
