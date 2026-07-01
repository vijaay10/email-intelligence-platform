"""
Analytics Dashboard Runner
---------------------------
Generates all charts and the text analytics report from whatever
email analysis data has been accumulated in the database.

Run:
    python scripts/run_analytics.py
    python scripts/run_analytics.py --db db/email_analysis.db --out reports/
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.database import DatabaseManager
from src.analytics import AnalyticsDashboard


def main():
    parser = argparse.ArgumentParser(description="Run email analytics dashboard.")
    parser.add_argument("--db",  default=None,      help="Path to SQLite DB.")
    parser.add_argument("--out", default="reports",  help="Output directory for charts.")
    parser.add_argument("--no-charts", action="store_true", help="Skip chart generation.")
    args = parser.parse_args()

    db = DatabaseManager(db_path=args.db)
    dash = AnalyticsDashboard(db, output_dir=args.out)

    stats = db.get_summary_stats()
    if stats["total_emails"] == 0:
        print("[Analytics] No emails in database yet. Run the analyser first.")
        return

    print(dash.generate_text_report())
    dash.save_text_report()

    if not args.no_charts:
        paths = dash.generate_all_charts()
        if paths:
            print(f"\n[Analytics] {len(paths)} chart(s) generated:")
            for p in paths:
                print(f"  {p}")
        else:
            print("[Analytics] No charts generated (insufficient data or matplotlib missing).")


if __name__ == "__main__":
    main()
