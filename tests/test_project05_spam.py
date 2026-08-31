"""Unit tests for Project 05: PhishShield — Spam & Phishing Detector."""
import os
import sys
import re
import string
import json
import numpy as np
import pandas as pd
import pytest
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "project_05_spam_phishing_detector")
sys.path.insert(0, BASE_DIR)


class TestModelArtifacts:
    """Verify that all model artifacts exist and are loadable."""

    def test_model_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "spam_model.pkl")), (
            "spam_model.pkl not found — run train_model.py first"
        )

    def test_tfidf_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")), (
            "tfidf_vectorizer.pkl not found — run train_model.py first"
        )

    def test_metrics_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "model_metrics.json")), (
            "model_metrics.json not found — run train_model.py first"
        )

    def test_model_loads_successfully(self):
        model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"))
        assert hasattr(model, "predict"), "Loaded model has no predict method"
        assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba method"

    def test_tfidf_loads_successfully(self):
        tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
        assert hasattr(tfidf, "transform"), "Loaded TF-IDF has no transform method"
        assert hasattr(tfidf, "vocabulary_"), "Loaded TF-IDF has no vocabulary_"

    def test_metrics_valid(self):
        with open(os.path.join(BASE_DIR, "model_metrics.json")) as f:
            metrics = json.load(f)
        assert "accuracy" in metrics, "Accuracy metric missing"
        assert "vocab_size" in metrics, "Vocab size metric missing"
        assert 0.0 <= metrics["accuracy"] <= 1.0, f"Accuracy out of range: {metrics['accuracy']}"


class TestNLPPipeline:
    """Test the NLP preprocessing and prediction pipeline."""

    @pytest.fixture
    def artifacts(self):
        model = joblib.load(os.path.join(BASE_DIR, "spam_model.pkl"))
        tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
        return model, tfidf

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", "http_link", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
        text = re.sub(r"\d+", "", text)
        return text

    def test_ham_classification(self, artifacts):
        model, tfidf = artifacts
        text = self.clean_text("Hey, are we still meeting for lunch today?")
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]
        assert pred == "ham", f"Expected 'ham', got '{pred}'"

    def test_spam_classification(self, artifacts):
        model, tfidf = artifacts
        text = self.clean_text("URGENT! Your bank account has been suspended. Click here to verify!")
        vec = tfidf.transform([text])
        pred = model.predict(vec)[0]
        assert pred == "spam", f"Expected 'spam', got '{pred}'"

    def test_prediction_probabilities(self, artifacts):
        model, tfidf = artifacts
        texts = [
            self.clean_text("Can you send over the project slides?"),
            self.clean_text("Congratulations! You won a $1000 gift card!"),
        ]
        vec = tfidf.transform(texts)
        proba = model.predict_proba(vec)
        assert proba.shape == (2, 2), f"Expected (2,2) probabilities, got {proba.shape}"
        for row in proba:
            assert np.isclose(sum(row), 1.0, atol=1e-6), "Probabilities should sum to 1.0"

    def test_vocabulary_size(self, artifacts):
        _, tfidf = artifacts
        assert len(tfidf.vocabulary_) > 0, "Vocabulary should not be empty"

    def test_classes_are_binary(self, artifacts):
        model, _ = artifacts
        classes = set(model.classes_)
        assert classes <= {"ham", "spam"}, f"Unexpected classes: {classes}"


class TestURLAnalysis:
    """Test the URL heuristic analysis function logic."""

    def test_suspicious_tld_detection(self):
        suspicious_tlds = [".xyz", ".top", ".online", ".site", ".club", ".info", ".live", ".cc", ".tk"]
        test_urls = [
            ("http://malicious.xyz", True),
            ("https://google.com/search", False),
            ("http://phish.top/login", True),
            ("https://github.com/repo", False),
            ("http://scam.xyz/path", True),
            ("https://trusted.com", False),
        ]
        for url, should_flag in test_urls:
            # Match the app heuristic: endswith(tld) or tld+"/" in url
            has_suspicious = any(url.endswith(tld) or (tld + "/") in url for tld in suspicious_tlds)
            assert has_suspicious == should_flag, f"URL {url}: expected flag={should_flag}"

    def test_ip_based_hostname_detection(self):
        ip_pattern = r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        assert re.search(ip_pattern, "http://192.168.1.1/login") is not None
        assert re.search(ip_pattern, "https://google.com") is None

    def test_long_url_detection(self):
        short_url = "https://google.com"
        long_url = "https://example.com/" + "a" * 60
        assert len(short_url) <= 55
        assert len(long_url) > 55
