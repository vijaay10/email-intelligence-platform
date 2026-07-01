"""
NLP Sentiment Model
-------------------
Core TF-IDF + Multinomial Naive Bayes classification engine.
Supports single prediction, batch prediction, and sentence-level analysis.
"""

import re
import os
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Optional NLTK stopwords — uses a built-in fallback set if not installed
_NLTK_SW_AVAILABLE = False
try:
    import nltk
    for _r in ["corpora/stopwords", "tokenizers/punkt"]:
        try:
            nltk.data.find(_r)
        except LookupError:
            nltk.download(_r.split("/")[-1], quiet=True)
    from nltk.corpus import stopwords as _nltk_stopwords
    STOPWORDS = set(_nltk_stopwords.words("english"))
    _NLTK_SW_AVAILABLE = True
except Exception:
    # Compact fallback stopword list — covers the most common English function words
    STOPWORDS = {
        "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
        "yourself","yourselves","he","him","his","himself","she","her","hers",
        "herself","it","its","itself","they","them","their","theirs","themselves",
        "what","which","who","whom","this","that","these","those","am","is","are",
        "was","were","be","been","being","have","has","had","having","do","does",
        "did","doing","a","an","the","and","but","if","or","because","as","until",
        "while","of","at","by","for","with","about","against","between","into",
        "through","during","before","after","above","below","to","from","up","down",
        "in","out","on","off","over","under","again","further","then","once","here",
        "there","when","where","why","how","all","both","each","few","more","most",
        "other","some","such","no","nor","not","only","own","same","so","than",
        "too","very","s","t","can","will","just","don","should","now","d","ll",
        "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
        "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn","weren",
        "won","wouldn",
    }


# ── SentimentModel ────────────────────────────────────────────────────────────
class SentimentModel:
    """
    TF-IDF vectoriser + Multinomial Naive Bayes classifier.
    Labels: 'positive' | 'neutral' | 'negative'
    """

    LABELS = ["negative", "neutral", "positive"]

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,          # dampen term frequency
        )
        self.classifier = MultinomialNB(alpha=0.5)
        self.is_trained: bool = False
        self.labels = self.LABELS

    # ── Text preprocessing ────────────────────────────────────────────────
    def preprocess_text(self, text: str) -> str:
        """Lowercase → strip URLs → strip non-alpha → remove stopwords."""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
        text = re.sub(r"[^a-zA-Z\s]", "", text)             # keep letters only
        text = re.sub(r"\s+", " ", text).strip()

        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        return " ".join(words)

    # ── Training ──────────────────────────────────────────────────────────
    def train(self, X_train, y_train) -> None:
        processed = [self.preprocess_text(t) for t in X_train]
        X_vec = self.vectorizer.fit_transform(processed)
        self.classifier.fit(X_vec, y_train)
        self.is_trained = True

    # ── Inference ─────────────────────────────────────────────────────────
    def _assert_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load_model() first.")

    def predict(self, text: str) -> dict:
        """Return sentiment label + confidence for a single text."""
        self._assert_trained()
        processed = self.preprocess_text(text)
        X_vec = self.vectorizer.transform([processed])
        label = self.classifier.predict(X_vec)[0]
        probs = self.classifier.predict_proba(X_vec)[0]
        return {
            "sentiment": label,
            "confidence": float(np.max(probs)),
            "probabilities": {
                lbl: float(p)
                for lbl, p in zip(self.classifier.classes_, probs)
            },
        }

    def predict_batch(self, texts: list) -> list:
        """Return list of {sentiment, confidence, probabilities} dicts."""
        self._assert_trained()
        processed = [self.preprocess_text(t) for t in texts]
        X_vec = self.vectorizer.transform(processed)
        labels = self.classifier.predict(X_vec)
        probs_matrix = self.classifier.predict_proba(X_vec)

        return [
            {
                "sentiment": lbl,
                "confidence": float(np.max(p)),
                "probabilities": {
                    cls: float(v)
                    for cls, v in zip(self.classifier.classes_, p)
                },
            }
            for lbl, p in zip(labels, probs_matrix)
        ]

    # ── Persistence ───────────────────────────────────────────────────────
    def save_model(self, filepath: str) -> None:
        payload = {
            "vectorizer": self.vectorizer,
            "classifier": self.classifier,
            "labels": self.labels,
            "is_trained": self.is_trained,
        }
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as fh:
            pickle.dump(payload, fh)
        print(f"[SentimentModel] Model saved → {filepath}")

    def load_model(self, filepath: str) -> None:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        with open(filepath, "rb") as fh:
            payload = pickle.load(fh)
        self.vectorizer = payload["vectorizer"]
        self.classifier = payload["classifier"]
        self.labels = payload["labels"]
        self.is_trained = payload["is_trained"]
        print(f"[SentimentModel] Model loaded ← {filepath}")
