# Demo Script (for the LinkedIn video / screen recording)

Target length: **60–90 seconds**. Record with macOS **Cmd + Shift + 5** →
"Record Selected Portion" around the browser window.

## Before you hit record
```bash
ollama serve                 # make sure the LLM daemon is up
cd "EmailAnalysisSystem 2" && ./run.sh ui
```
Have a few emails already analyzed (demo mode or a real fetch) so the pages
aren't empty:
```bash
printf '2\nQ\n' | .venv/bin/python main.py     # loads 5 demo emails
.venv/bin/python scripts/backfill_embeddings.py
```

## The 6 beats (≈15s each)

1. **Home (5s)** — show the title + status panel: "LLM available · N vectors ·
   ML pipeline". Say: *"A local-first email assistant — ML + LLM + RAG, no cloud."*

2. **AI Summary (20s)** — pick the "Server Outage" email → **Analyze with AI**.
   Let the summary, action items, risk (HIGH), and the **"Why the ML rated this
   HIGH"** explanation appear. This is the money shot — ML + LLM together.

3. **Semantic Search (10s)** — type *"which invoices are overdue?"* → the invoice
   email ranks #1. Say: *"Search by meaning, not keywords."*

4. **Chat with Emails (20s)** — ask *"Which emails need urgent action?"* → grounded
   answer with sources. Say: *"It answers only from my actual emails."*

5. **Risk Dashboard (10s)** — click **Assess risk** → the red/amber/green metrics
   and flagged list.

6. **Close (5s)** — back to Home; say *"100% local — no email ever leaves my
   machine."* End on the GitHub URL.

## Caption ideas
- "Turning a plain inbox into an AI assistant — running entirely on my laptop."
- Pin the repo link as the first comment.
