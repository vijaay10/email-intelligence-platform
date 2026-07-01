"""
Training Script  (Feature 6 — Large Dataset Support)
------------------------------------------------------
Supports:
  - Standard CSV  (columns: text, label)
  - Large CSVs via chunked streaming
  - Directory of .txt files (label inferred from subfolder name)
  - Merging multiple dataset sources

Run:
    python scripts/train_model.py
    python scripts/train_model.py --data data/my_large_dataset.csv --chunk-size 50000
    python scripts/train_model.py --data-dir data/email_corpus/
"""

import os
import sys
import argparse
import logging
import datetime

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.utils import shuffle as sk_shuffle

# ── path setup so we can import src.* when run as a script ─────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.nlp_model import SentimentModel

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ── defaults ─────────────────────────────────────────────────────────────────
DEFAULT_DATA_PATH   = os.path.join(ROOT, "data", "train_data.csv")
DEFAULT_MODEL_PATH  = os.path.join(ROOT, "models", "sentiment_model.pkl")
DEFAULT_METRICS_PATH = os.path.join(ROOT, "reports", "training_metrics.txt")
DEFAULT_CHUNK_SIZE  = 100_000     # rows per chunk for large files
VALID_LABELS        = {"positive", "neutral", "negative"}


# ── Data loaders ──────────────────────────────────────────────────────────────

def load_csv(path: str, chunk_size: int = None) -> pd.DataFrame:
    """
    Load a CSV with ``text`` and ``label`` columns.
    For large files, reads in chunks to limit peak memory.
    """
    logger.info("Loading CSV: %s", path)

    # Auto-detect delimiter
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        sample = fh.read(2048)
    sep = "," if sample.count(",") >= sample.count("\t") else "\t"

    total_rows = sum(1 for _ in open(path, encoding="utf-8", errors="ignore")) - 1
    logger.info("  Rows in file: %d", total_rows)

    if chunk_size and total_rows > chunk_size:
        logger.info("  Streaming in chunks of %d…", chunk_size)
        chunks = []
        for i, chunk in enumerate(
            pd.read_csv(path, sep=sep, chunksize=chunk_size,
                        encoding="utf-8", on_bad_lines="skip")
        ):
            chunks.append(_validate_chunk(chunk))
            logger.info("  Chunk %d loaded (%d rows)", i + 1, len(chunk))
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(path, sep=sep, encoding="utf-8", on_bad_lines="skip")
        df = _validate_chunk(df)

    logger.info("  Total loaded: %d rows", len(df))
    return df


def _validate_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise column names, drop invalid labels and NaN."""
    df.columns = [c.lower().strip() for c in df.columns]

    # Try to auto-map common column names
    col_map = {}
    for col in df.columns:
        if col in ("text", "content", "email", "message", "body"):
            col_map[col] = "text"
        elif col in ("label", "sentiment", "class", "category"):
            col_map[col] = "label"
    df.rename(columns=col_map, inplace=True)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(
            f"Dataset must contain 'text' and 'label' columns. "
            f"Found: {list(df.columns)}"
        )

    df = df[["text", "label"]].dropna()
    df["label"] = df["label"].str.lower().str.strip()
    df = df[df["label"].isin(VALID_LABELS)]
    df = df[df["text"].str.strip().str.len() > 3]
    return df.reset_index(drop=True)


def load_text_directory(directory: str) -> pd.DataFrame:
    """
    Load emails from a directory structure:
        directory/positive/*.txt
        directory/negative/*.txt
        directory/neutral/*.txt
    """
    records = []
    for label in VALID_LABELS:
        label_dir = os.path.join(directory, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            if fname.endswith(".txt"):
                fpath = os.path.join(label_dir, fname)
                try:
                    with open(fpath, encoding="utf-8", errors="ignore") as fh:
                        text = fh.read().strip()
                    if text:
                        records.append({"text": text, "label": label})
                except Exception as exc:
                    logger.warning("Could not read %s: %s", fpath, exc)

    df = pd.DataFrame(records)
    logger.info("Loaded %d records from directory %s", len(df), directory)
    return df


def merge_datasets(*dfs: pd.DataFrame) -> pd.DataFrame:
    """Concatenate multiple DataFrames, shuffle and deduplicate."""
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.drop_duplicates(subset="text")
    merged = sk_shuffle(merged, random_state=42).reset_index(drop=True)
    return merged


# ── Training pipeline ─────────────────────────────────────────────────────────

def train_and_evaluate(
    data_path: str = DEFAULT_DATA_PATH,
    model_path: str = DEFAULT_MODEL_PATH,
    metrics_path: str = DEFAULT_METRICS_PATH,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    data_dir: str = None,
    test_size: float = 0.20,
    max_features: int = 5000,
):
    """
    Full training pipeline.  Loads data, trains the model, evaluates,
    saves model + metrics, returns the trained model and metrics dict.
    """
    print("=" * 62)
    print("  SENTIMENT MODEL — TRAINING PIPELINE")
    print(f"  {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 62)

    os.makedirs(os.path.dirname(model_path),  exist_ok=True)
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    # ── 1. Load data ──────────────────────────────────────────────────────
    print("\n[1/8] Loading data…")
    dfs = []
    if data_dir and os.path.isdir(data_dir):
        dfs.append(load_text_directory(data_dir))
    if os.path.exists(data_path):
        dfs.append(load_csv(data_path, chunk_size=chunk_size))

    if not dfs:
        raise FileNotFoundError(
            f"No training data found at {data_path} or {data_dir}"
        )

    df = merge_datasets(*dfs) if len(dfs) > 1 else dfs[0]
    print(f"  Total samples : {len(df)}")
    print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")

    # ── 2. Split ──────────────────────────────────────────────────────────
    print(f"\n[2/8] Splitting {(1-test_size)*100:.0f}% train / {test_size*100:.0f}% test…")
    # Use plain Python lists: pandas >= 3.0 returns PyArrow-backed arrays from
    # ``.values``, which scikit-learn's train_test_split cannot index.
    texts = df["text"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    # ── 3. Train ──────────────────────────────────────────────────────────
    print("\n[3/8] Training model (TF-IDF + Multinomial Naive Bayes)…")
    model = SentimentModel(max_features=max_features)
    model.train(X_train, y_train)
    print("  ✓  Training complete.")

    # ── 4. Evaluate (train set) ───────────────────────────────────────────
    print("\n[4/8] Evaluating on train set…")
    train_preds = [p["sentiment"] for p in model.predict_batch(X_train)]
    train_acc = accuracy_score(y_train, train_preds)
    print(f"  Train Accuracy: {train_acc:.4f}")

    # ── 5. Evaluate (test set) ────────────────────────────────────────────
    print("\n[5/8] Evaluating on test set…")
    test_preds = [p["sentiment"] for p in model.predict_batch(X_test)]
    test_acc = accuracy_score(y_test, test_preds)
    print(f"  Test Accuracy : {test_acc:.4f}")

    # ── 6. Classification report ──────────────────────────────────────────
    print("\n[6/8] Classification report:")
    report = classification_report(y_test, test_preds)
    print(report)

    # ── 7. Confusion matrix ───────────────────────────────────────────────
    print("\n[7/8] Confusion matrix:")
    labels_order = ["positive", "neutral", "negative"]
    cm = confusion_matrix(y_test, test_preds, labels=labels_order)
    header = "              " + "  ".join(f"{l[:3]:>5}" for l in labels_order)
    print(header)
    for row_label, row in zip(labels_order, cm):
        print(f"  {row_label:<10}  " + "  ".join(f"{v:>5}" for v in row))

    # ── 8. Save ───────────────────────────────────────────────────────────
    print(f"\n[8/8] Saving model → {model_path}")
    model.save_model(model_path)

    metrics = {
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "total_samples": len(df),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }
    _write_metrics(metrics, metrics_path, df)

    print("\n" + "=" * 62)
    print(f"  TRAINING COMPLETE!  Test accuracy: {test_acc:.4f}")
    print("=" * 62 + "\n")
    return model, metrics


def _write_metrics(metrics: dict, path: str, df: pd.DataFrame):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("=" * 62 + "\n")
        fh.write("SENTIMENT MODEL — METRICS REPORT\n")
        fh.write(f"Generated: {datetime.datetime.now()}\n")
        fh.write("=" * 62 + "\n\n")
        fh.write(f"Total samples : {metrics['total_samples']}\n")
        fh.write(f"Train samples : {metrics['train_samples']}\n")
        fh.write(f"Test samples  : {metrics['test_samples']}\n\n")
        fh.write("Label distribution:\n")
        fh.write(df["label"].value_counts().to_string() + "\n\n")
        fh.write(f"Train accuracy : {metrics['train_accuracy']:.4f}\n")
        fh.write(f"Test accuracy  : {metrics['test_accuracy']:.4f}\n\n")
        fh.write("Classification Report:\n")
        fh.write(metrics["classification_report"] + "\n")
    print(f"  Metrics saved → {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli():
    parser = argparse.ArgumentParser(
        description="Train the email sentiment model."
    )
    parser.add_argument(
        "--data", default=DEFAULT_DATA_PATH,
        help="Path to training CSV (text, label columns)."
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory of labelled .txt files (positive/neutral/negative sub-folders)."
    )
    parser.add_argument(
        "--model-out", default=DEFAULT_MODEL_PATH,
        help="Where to save the trained model."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help="Rows per chunk for large CSV files."
    )
    parser.add_argument(
        "--max-features", type=int, default=5000,
        help="TF-IDF vocabulary size."
    )
    parser.add_argument(
        "--test-size", type=float, default=0.20,
        help="Fraction of data to hold out for testing."
    )
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data,
        model_path=args.model_out,
        chunk_size=args.chunk_size,
        data_dir=args.data_dir,
        test_size=args.test_size,
        max_features=args.max_features,
    )


if __name__ == "__main__":
    _cli()
