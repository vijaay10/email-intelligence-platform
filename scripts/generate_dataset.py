"""
Training Dataset Generator
==========================
Builds a larger, balanced, varied sentiment dataset for the ML model. The
bundled ``train_data.csv`` has only ~50 rows, which is far too small for
real-world use (the neutral class never learns). This produces a few hundred
diverse, email-style examples across positive / neutral / negative.

Run:
    python scripts/generate_dataset.py                       # → data/train_data_large.csv
    python scripts/generate_dataset.py --rows 1200 --out data/my.csv

NOTE: This is *synthetic augmentation* — good enough to make the demo behave
sensibly and to show a proper train/eval. For production, replace or extend it
with a real labeled email corpus (same two columns: text,label).
"""

import os
import csv
import sys
import random
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Vocabulary banks ───────────────────────────────────────────────────────────
SUBJECTS = ["the team", "our vendor", "the client", "support", "the project",
            "the release", "your service", "the delivery", "the report",
            "the meeting", "the invoice", "the product", "the update", "the demo"]

POS_ADJ = ["excellent", "fantastic", "outstanding", "wonderful", "great",
           "impressive", "smooth", "delightful", "superb", "flawless"]
NEG_ADJ = ["terrible", "unacceptable", "broken", "disappointing", "awful",
           "frustrating", "critical", "urgent", "faulty", "delayed"]

POS_TEMPLATES = [
    "I love how {subj} turned out, it was {adj}.",
    "Thank you so much, {subj} was absolutely {adj}.",
    "Great job on {subj} — the results were {adj} and exceeded expectations.",
    "We are very happy with {subj}, truly {adj} work.",
    "Congratulations to everyone, {subj} looked {adj} today.",
    "Really appreciate the {adj} effort on {subj}.",
    "{subj} was {adj}; the client was thrilled with the outcome.",
    "Fantastic news about {subj}, everything went {adj}.",
]
NEG_TEMPLATES = [
    "I am extremely disappointed with {subj}, it was {adj}.",
    "This is {adj}. {subj} failed again and must be fixed immediately.",
    "We have a {adj} problem with {subj} and need it resolved asap.",
    "{subj} is completely {adj}; our clients are complaining.",
    "Urgent: {subj} is {adj} and we risk losing the contract.",
    "The {adj} issue with {subj} has not been addressed despite reminders.",
    "I demand a refund — {subj} was {adj} and unusable.",
    "Escalating this: {subj} is {adj} and downtime is increasing.",
]
NEU_TEMPLATES = [
    "Please find attached the notes for {subj}.",
    "{subj} is scheduled for Thursday at 10:00 AM.",
    "Here is the weekly status update on {subj}.",
    "Can you confirm the details for {subj} before Friday?",
    "The agenda for {subj} covers scope, timeline, and budget.",
    "We reviewed {subj} and there are no changes to report.",
    "Forwarding the documentation related to {subj} for your reference.",
    "Reminder: {subj} takes place next week as planned.",
]

TEMPLATES = {"positive": (POS_TEMPLATES, POS_ADJ),
             "negative": (NEG_TEMPLATES, NEG_ADJ),
             "neutral":  (NEU_TEMPLATES, ["scheduled", "planned", "pending",
                                          "documented", "confirmed", "routine"])}


def generate(rows_per_class: int, seed: int = 42):
    random.seed(seed)
    out = set()
    for label, (templates, adjectives) in TEMPLATES.items():
        attempts = 0
        made = 0
        while made < rows_per_class and attempts < rows_per_class * 40:
            attempts += 1
            text = random.choice(templates).format(
                subj=random.choice(SUBJECTS), adj=random.choice(adjectives))
            key = (text, label)
            if key not in out:
                out.add(key)
                made += 1
    rows = list(out)
    random.shuffle(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a balanced sentiment CSV.")
    parser.add_argument("--rows", type=int, default=750,
                        help="Approx total rows (split evenly across 3 classes).")
    parser.add_argument("--out", default=os.path.join(ROOT, "data", "train_data_large.csv"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    per_class = max(1, args.rows // 3)
    rows = generate(per_class, seed=args.seed)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(rows)

    counts = {}
    for _, label in rows:
        counts[label] = counts.get(label, 0) + 1
    print(f"Wrote {len(rows)} rows → {args.out}")
    print(f"Label distribution: {counts}")


if __name__ == "__main__":
    main()
