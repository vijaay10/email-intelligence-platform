"""
Chat with Emails — CLI  (Phase 3)
=================================
Interactive REPL that answers questions about your indexed emails using RAG +
a local LLM (Ollama).

Prerequisites:
    1. pip install -r requirements.txt
    2. ollama serve   (and: ollama pull llama3)
    3. python scripts/backfill_embeddings.py   (to index your emails)

Run:
    python scripts/chat.py
    python scripts/chat.py --conversation work

Type 'exit' or Ctrl+C to quit.
"""

import os
import sys
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.services.chat_service import ChatService


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with your emails (RAG + LLM).")
    parser.add_argument("--conversation", default="default",
                        help="Conversation id (lets you keep separate threads).")
    args = parser.parse_args()

    chat = ChatService()

    print("=" * 62)
    print("  CHAT WITH EMAILS")
    print("=" * 62)

    if not chat.is_available():
        print(f"\n[!] LLM unavailable: {chat.unavailable_reason()}")
        print("    Start Ollama:  ollama serve   (then: ollama pull llama3)")
        return
    if not chat.rag.is_available():
        print(f"\n[i] Note: semantic search is unavailable "
              f"({chat.rag.unavailable_reason()}).")
        print("    Answers will not be grounded in your emails until you install")
        print("    the RAG stack and run scripts/backfill_embeddings.py.\n")

    print("\n  Ask a question (type 'exit' to quit).\n")
    try:
        while True:
            question = input("  you > ").strip()
            if question.lower() in ("exit", "quit", "q"):
                break
            if not question:
                continue

            result = chat.ask(question, conversation_id=args.conversation)
            print(f"\n  bot > {result['answer']}\n")

            if result.get("sources"):
                print("  sources:")
                for s in result["sources"]:
                    print(f"    - {s.get('sender', '')} | {s.get('subject', '')} "
                          f"(score={s.get('score')})")
                print()
    except (KeyboardInterrupt, EOFError):
        pass
    print("\n  Goodbye!\n")


if __name__ == "__main__":
    main()
