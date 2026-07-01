"""
Demo Reel Generator
===================
Builds an animated GIF showcasing the platform using its REAL outputs — a live
llama3 summary, real semantic-search ranking, a grounded chat answer, and the
analytics chart. Produces docs/demo.gif for the README / LinkedIn.

Run:
    python scripts/make_demo_reel.py

Falls back to placeholder text for any feature whose backend is unavailable,
so it never crashes.
"""

import os
import sys
import textwrap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import config
from src.database import DatabaseManager
from src.services.llm_service import LlmService
from src.services.rag_service import RagService
from src.services.chat_service import ChatService

# ── Palette ────────────────────────────────────────────────────────────────────
BG = "#0E1117"
FG = "#FAFAFA"
ACCENT = "#7C4DFF"
GREEN = "#4CAF50"
RED = "#F44336"
W, H, DPI = 1000, 563, 100  # 16:9


def _fig():
    fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI, facecolor=BG)
    return fig


def _to_image(fig) -> Image.Image:
    fig.canvas.draw()
    img = Image.frombytes("RGBA", fig.canvas.get_width_height(),
                          bytes(fig.canvas.buffer_rgba())).convert("RGB")
    plt.close(fig)
    return img


def slide_text(kicker, title, body_lines, accent=ACCENT):
    fig = _fig()
    if len(title) > 38:
        title = title[:37].rstrip() + "…"
    fig.text(0.07, 0.86, kicker, color=accent, fontsize=15, fontweight="bold")
    fig.text(0.07, 0.76, title, color=FG, fontsize=24, fontweight="bold")
    y = 0.62
    for line in body_lines:
        wrapped = textwrap.wrap(line, width=72) or [""]
        for w in wrapped:
            fig.text(0.07, y, w, color=FG, fontsize=15)
            y -= 0.075
        y -= 0.02
    fig.text(0.07, 0.06, "Email Intelligence Platform · 100% local (Ollama)",
             color="#888", fontsize=11)
    return _to_image(fig)


def slide_title():
    fig = _fig()
    fig.text(0.5, 0.60, "📧  Email Intelligence Platform", color=FG,
             fontsize=30, fontweight="bold", ha="center")
    fig.text(0.5, 0.48, "Hybrid  ML + LLM + RAG  ·  runs entirely on your machine",
             color=ACCENT, fontsize=17, ha="center")
    fig.text(0.5, 0.30, "TF-IDF + Naive Bayes   ·   Llama 3 (Ollama)   ·   ChromaDB   ·   Streamlit",
             color="#AAA", fontsize=12, ha="center")
    return _to_image(fig)


def slide_chart(path, title):
    fig = _fig()
    fig.text(0.07, 0.90, title, color=ACCENT, fontsize=15, fontweight="bold")
    if os.path.exists(path):
        ax = fig.add_axes([0.05, 0.05, 0.9, 0.78])
        ax.imshow(np.array(Image.open(path)))
        ax.axis("off")
    return _to_image(fig)


def slide_end():
    fig = _fig()
    fig.text(0.5, 0.58, "🔒  No email ever leaves your machine", color=GREEN,
             fontsize=22, fontweight="bold", ha="center")
    fig.text(0.5, 0.42, "github.com/vijaay10/email-intelligence-platform",
             color=FG, fontsize=16, ha="center")
    return _to_image(fig)


def main() -> None:
    db = DatabaseManager()
    llm = LlmService(db=db)
    rag = RagService(db=db)
    chat = ChatService(rag=rag, db=db)

    emails = db.get_all_emails()
    outage = next((e for e in emails if "Outage" in (e.get("subject") or "")),
                  emails[0] if emails else None)

    frames = [slide_title()]

    # 1. AI Summary (real llama3 output)
    if outage and llm.is_available():
        s = llm.summarize(outage, outage["id"]).get("summary", "")
        cat = llm.categorize(outage, outage["id"]).get("category", "")
        risk = llm.detect_risks(outage, outage["id"])
        frames.append(slide_text(
            "AI SUMMARY", outage.get("subject", "")[:46],
            [s, "", f"Category: {cat}    Risk: {risk.get('risk_level','').upper()}  "
                    f"({', '.join(risk.get('risks', [])) or 'none'})"]))
    else:
        frames.append(slide_text("AI SUMMARY", "LLM summary",
                                  ["(LLM unavailable when this reel was built.)"]))

    # 2. Semantic Search (real ranking)
    if rag.is_available():
        hits = rag.search("which invoices are overdue?", top_k=3)
        lines = [f"→ {h['metadata'].get('subject','')[:50]}  (score {h.get('score')})"
                 for h in hits]
        frames.append(slide_text("SEMANTIC SEARCH",
                                  "“which invoices are overdue?”",
                                  lines or ["(no results)"]))
    else:
        frames.append(slide_text("SEMANTIC SEARCH", "Search by meaning",
                                  ["(RAG unavailable when this reel was built.)"]))

    # 3. Chat with Emails (real grounded answer)
    if chat.is_available():
        ans = chat.ask("Which emails need urgent action?", conversation_id="reel")
        frames.append(slide_text("CHAT WITH EMAILS",
                                  "“Which emails need urgent action?”",
                                  [ans.get("answer", "")[:300]]))
    else:
        frames.append(slide_text("CHAT WITH EMAILS", "Ask your inbox",
                                  ["(LLM unavailable when this reel was built.)"]))

    # 4. Analytics chart
    frames.append(slide_chart(os.path.join(ROOT, "reports", "summary_dashboard.png"),
                              "ANALYTICS  ·  from the ML pipeline"))

    frames.append(slide_end())

    out = os.path.join(ROOT, "docs", "demo.gif")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=2800, loop=0, optimize=True)
    print(f"Wrote {out}  ({len(frames)} frames)")


if __name__ == "__main__":
    main()
