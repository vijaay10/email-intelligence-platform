"""
Unit Tests — Advanced Intelligent Email Analysis System
--------------------------------------------------------
Run:
    python -m pytest tests/ -v
    python tests/test_all.py        # standalone
"""

import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# The ML model needs scikit-learn + numpy. When they are absent (minimal
# install) we skip those tests instead of erroring, keeping the suite green.
try:
    import sklearn  # noqa: F401
    import numpy    # noqa: F401
    _ML_AVAILABLE = True
except Exception:
    _ML_AVAILABLE = False


# ── Test: ImportancePredictor ─────────────────────────────────────────────────
class TestImportancePredictor(unittest.TestCase):

    def setUp(self):
        from src.importance_predictor import ImportancePredictor
        self.predictor = ImportancePredictor()

    def test_high_importance(self):
        result = self.predictor.calculate_importance(
            email_text="URGENT! We have a critical problem. asap deadline failure!!",
            sender="manager@corp.com",
            sentiment="negative",
        )
        self.assertEqual(result.importance_level, "HIGH")
        self.assertGreater(result.total_score, 6)

    def test_low_importance(self):
        result = self.predictor.calculate_importance(
            email_text="Hi",
            sender="unknown@mail.com",
            sentiment="positive",
        )
        self.assertEqual(result.importance_level, "LOW")
        self.assertLess(result.total_score, 3)

    def test_urgency_keywords_detected(self):
        result = self.predictor.calculate_importance(
            email_text="There is an urgent issue with the deadline.",
            sender="x@y.com",
            sentiment="neutral",
        )
        self.assertIn("urgent", result.urgency_keywords_found)
        self.assertIn("issue", result.urgency_keywords_found)
        self.assertIn("deadline", result.urgency_keywords_found)

    def test_negative_sentiment_raises_score(self):
        pos = self.predictor.calculate_importance("Hello", "a@b.com", "positive")
        neg = self.predictor.calculate_importance("Hello", "a@b.com", "negative")
        self.assertGreater(neg.total_score, pos.total_score)

    def test_sender_role_matched(self):
        result = self.predictor.calculate_importance(
            "meeting tomorrow", "boss@company.com", "neutral"
        )
        self.assertEqual(result.sender_role_detected, "boss")
        self.assertEqual(result.sender_score, 2.0)

    def test_exclamation_score(self):
        result = self.predictor.calculate_importance(
            "Wow!!!!!!", "a@b.com", "neutral"
        )
        self.assertEqual(result.exclamation_score, 2.0)

    def test_importance_result_to_dict(self):
        result = self.predictor.calculate_importance("test", "a@b.com", "neutral")
        d = result.to_dict()
        self.assertIn("importance_level", d)
        self.assertIn("total_score", d)
        self.assertIn("score_breakdown", d)


# ── Test: SentenceAnalyzer ────────────────────────────────────────────────────
class TestSentenceAnalyzer(unittest.TestCase):

    def _make_mock_model(self):
        """Minimal mock model that classifies by keyword heuristic."""
        class MockModel:
            is_trained = True

            def predict_batch(self, texts):
                results = []
                for t in texts:
                    tl = t.lower()
                    if any(w in tl for w in ["great", "happy", "love", "excellent"]):
                        s = "positive"
                    elif any(w in tl for w in ["bad", "urgent", "fail", "problem", "terrible"]):
                        s = "negative"
                    else:
                        s = "neutral"
                    results.append({"sentiment": s, "confidence": 0.90, "probabilities": {}})
                return results

        return MockModel()

    def setUp(self):
        from src.sentence_analyzer import SentenceAnalyzer
        self.analyser = SentenceAnalyzer(self._make_mock_model())

    def test_single_sentence(self):
        result = self.analyser.analyse("This is great news!")
        self.assertEqual(result["sentence_count"], 1)
        self.assertFalse(result["sentiment_shift"])

    def test_shift_detected(self):
        text = "This is great! I love it. But there is a terrible problem. The system failed."
        result = self.analyser.analyse(text)
        self.assertTrue(result["sentiment_shift"])

    def test_no_shift_uniform_positive(self):
        text = "Excellent work. Great results. Love this!"
        result = self.analyser.analyse(text)
        self.assertFalse(result["sentiment_shift"])
        self.assertEqual(result["overall_sentiment"], "positive")

    def test_empty_body(self):
        result = self.analyser.analyse("")
        self.assertEqual(result["sentence_count"], 0)
        self.assertEqual(result["overall_sentiment"], "neutral")

    def test_sentences_have_required_keys(self):
        result = self.analyser.analyse("Hello world. How are you?")
        for s in result["sentences"]:
            self.assertIn("sentence", s)
            self.assertIn("sentiment", s)
            self.assertIn("confidence", s)
            self.assertIn("index", s)


# ── Test: split_into_sentences ────────────────────────────────────────────────
class TestSentenceSplitter(unittest.TestCase):

    def test_basic_split(self):
        from src.sentence_analyzer import split_into_sentences
        sents = split_into_sentences("Hello. My name is Claude. How are you?")
        self.assertGreaterEqual(len(sents), 2)

    def test_empty(self):
        from src.sentence_analyzer import split_into_sentences
        self.assertEqual(split_into_sentences(""), [])
        self.assertEqual(split_into_sentences(None), [])

    def test_single(self):
        from src.sentence_analyzer import split_into_sentences
        result = split_into_sentences("Just one sentence here")
        self.assertEqual(len(result), 1)


# ── Test: DatabaseManager ─────────────────────────────────────────────────────
class TestDatabaseManager(unittest.TestCase):

    def setUp(self):
        from src.database import DatabaseManager
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db = DatabaseManager(db_path=self.tmp.name)

    def tearDown(self):
        os.unlink(self.tmp.name)

    def _insert_sample(self, sentiment="positive", importance="LOW"):
        return self.db.save_email_analysis(
            sender="test@example.com",
            subject="Test Subject",
            email_body="This is a test email body.",
            overall_sentiment=sentiment,
            importance_level=importance,
            sentiment_shift=False,
            urgency_score=1.5,
            date="2025-10-01",
            sentences=[
                {"index": 1, "sentence": "This is a test.", "sentiment": sentiment, "confidence": 0.95},
            ],
        )

    def test_insert_and_retrieve(self):
        eid = self._insert_sample()
        self.assertIsInstance(eid, int)
        self.assertGreater(eid, 0)

        record = self.db.get_email_by_id(eid)
        self.assertIsNotNone(record)
        self.assertEqual(record["sender"], "test@example.com")

    def test_sentences_stored(self):
        eid = self._insert_sample()
        sents = self.db.get_sentences_for_email(eid)
        self.assertEqual(len(sents), 1)
        self.assertEqual(sents[0]["sentiment"], "positive")

    def test_sentiment_counts(self):
        self._insert_sample("positive")
        self._insert_sample("negative")
        self._insert_sample("negative")
        counts = self.db.get_sentiment_counts()
        self.assertEqual(counts.get("positive", 0), 1)
        self.assertEqual(counts.get("negative", 0), 2)

    def test_importance_counts(self):
        self._insert_sample(importance="HIGH")
        self._insert_sample(importance="LOW")
        counts = self.db.get_importance_counts()
        self.assertIn("HIGH", counts)
        self.assertIn("LOW", counts)

    def test_most_negative_senders(self):
        self._insert_sample("negative")
        results = self.db.get_most_negative_senders()
        self.assertTrue(len(results) > 0)

    def test_summary_stats(self):
        self._insert_sample("negative", "HIGH")
        stats = self.db.get_summary_stats()
        self.assertIn("total_emails", stats)
        self.assertIn("high_importance", stats)
        self.assertGreaterEqual(stats["total_emails"], 1)

    def test_get_all_emails(self):
        self._insert_sample()
        self._insert_sample()
        all_emails = self.db.get_all_emails()
        self.assertGreaterEqual(len(all_emails), 2)

    def test_reset(self):
        self._insert_sample()
        self.db.reset_database()
        self.assertEqual(len(self.db.get_all_emails()), 0)


# ── Test: NLPModel preprocessing ─────────────────────────────────────────────
@unittest.skipUnless(_ML_AVAILABLE, "scikit-learn/numpy not installed")
class TestNLPModel(unittest.TestCase):

    def setUp(self):
        from src.nlp_model import SentimentModel
        self.model = SentimentModel()

    def test_preprocess_removes_urls(self):
        result = self.model.preprocess_text("visit http://example.com now")
        self.assertNotIn("http", result)

    def test_preprocess_removes_numbers(self):
        result = self.model.preprocess_text("call 123-456-7890 today")
        self.assertNotIn("123", result)

    def test_preprocess_lowercase(self):
        result = self.model.preprocess_text("Hello World")
        self.assertEqual(result, result.lower())

    def test_predict_requires_training(self):
        with self.assertRaises(RuntimeError):
            self.model.predict("some text")

    def test_train_and_predict(self):
        texts  = ["I love this", "I hate this", "It is okay"] * 10
        labels = ["positive",    "negative",    "neutral"   ] * 10
        self.model.train(texts, labels)
        result = self.model.predict("I love this")
        self.assertIn(result["sentiment"], ["positive", "neutral", "negative"])
        self.assertGreater(result["confidence"], 0)

    def test_save_and_load(self):
        texts  = ["great", "terrible", "fine"] * 10
        labels = ["positive", "negative", "neutral"] * 10
        self.model.train(texts, labels)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            path = f.name
        try:
            self.model.save_model(path)
            from src.nlp_model import SentimentModel
            loaded = SentimentModel()
            loaded.load_model(path)
            result = loaded.predict("great day")
            self.assertIn(result["sentiment"], ["positive", "neutral", "negative"])
        finally:
            os.unlink(path)


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
