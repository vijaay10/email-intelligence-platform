"""
Embedding Backfill / Semantic Search CLI  (Phase 1 — RAG)
=========================================================
Embeds every email currently stored in the SQLite database into the ChromaDB
vector store, skipping any whose content is unchanged. Optionally runs a
semantic search afterward.

Run:
    # Embed all stored emails (incremental — skips unchanged):
    python scripts/backfill_embeddings.py

    # Force re-embed everything:
    python scripts/backfill_embeddings.py --force

    # Embed, then run a semantic search:
    python scripts/backfill_embeddings.py --search "which invoices are overdue?"

Requires ``sentence-transformers`` and ``chromadb`` (see requirements.txt).
If they are not installed the script reports that clearly and exits cleanly.
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.services.rag_service import RagService


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill email embeddings (RAG).")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed every email even if unchanged.")
    parser.add_argument("--search", default=None, metavar="QUERY",
                        help="Run a semantic search after backfilling.")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of results for --search.")
    args = parser.parse_args()

    rag = RagService()

    print("=" * 62)
    print("  RAG — EMBEDDING BACKFILL")
    print("=" * 62)

    if not rag.is_available():
        print(f"\n[!] RAG backend unavailable: {rag.unavailable_reason()}")
        print("    Install the AI stack:  pip install -r requirements.txt")
        return

    stats = rag.backfill_from_db(force=args.force)
    print(f"\n  Total emails : {stats['total']}")
    print(f"  Embedded     : {stats['embedded']}")
    print(f"  Skipped      : {stats['skipped']} (unchanged)")
    print(f"  Vectors now  : {rag.store.count()}")

    if args.search:
        print("\n" + "=" * 62)
        print(f"  SEMANTIC SEARCH:  {args.search!r}")
        print("=" * 62)
        results = rag.search(args.search, top_k=args.top_k)
        if not results:
            print("  No results.")
        for i, hit in enumerate(results, 1):
            meta = hit["metadata"]
            print(f"\n  {i}. score={hit.get('score')}  "
                  f"from={meta.get('sender', '')}")
            print(f"     subject: {meta.get('subject', '')}")
            print(f"     {hit['document'][:120].replace(chr(10), ' ')}…")


if __name__ == "__main__":
    main()
