"""Unit tests for Project 04: ValuaAI — Real Estate Price Estimator."""
import os
import sys
import json
import numpy as np
import pandas as pd
import pytest
import joblib

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "project_04_real_estate_ai")
sys.path.insert(0, BASE_DIR)


class TestModelArtifacts:
    """Verify that all model artifacts exist and are loadable."""

    def test_model_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "real_estate_model.pkl")), (
            "real_estate_model.pkl not found — run train_model.py first"
        )

    def test_scaler_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "real_estate_scaler.pkl")), (
            "real_estate_scaler.pkl not found — run train_model.py first"
        )

    def test_metrics_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "model_metrics.json")), (
            "model_metrics.json not found — run train_model.py first"
        )

    def test_model_loads_successfully(self):
        model = joblib.load(os.path.join(BASE_DIR, "real_estate_model.pkl"))
        assert hasattr(model, "predict"), "Loaded object has no predict method"
        assert hasattr(model, "feature_importances_"), "Loaded model has no feature_importances_"

    def test_scaler_loads_successfully(self):
        scaler = joblib.load(os.path.join(BASE_DIR, "real_estate_scaler.pkl"))
        assert hasattr(scaler, "transform"), "Loaded scaler has no transform method"

    def test_metrics_valid(self):
        with open(os.path.join(BASE_DIR, "model_metrics.json")) as f:
            metrics = json.load(f)
        assert "mae" in metrics, "MAE metric missing"
        assert "r2" in metrics, "R2 metric missing"
        assert metrics["r2"] > 0.9, f"R2 score too low: {metrics['r2']}"
        assert metrics["mae"] > 0, "MAE should be positive"


class TestPredictionPipeline:
    """Test that the model makes valid predictions."""

    @pytest.fixture
    def artifacts(self):
        model = joblib.load(os.path.join(BASE_DIR, "real_estate_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "real_estate_scaler.pkl"))
        return model, scaler

    def test_single_prediction(self, artifacts):
        model, scaler = artifacts
        sample = pd.DataFrame([{
            "sqft": 2200, "bedrooms": 3, "bathrooms": 2,
            "age": 5, "location_score": 7,
        }])
        scaled = scaler.transform(sample)
        pred = model.predict(scaled)
        assert pred.shape == (1,), f"Expected 1 prediction, got {pred.shape}"
        assert pred[0] > 0, f"Prediction should be positive, got {pred[0]}"

    def test_batch_prediction(self, artifacts):
        model, scaler = artifacts
        sample = pd.DataFrame({
            "sqft": [1000, 2000, 4000],
            "bedrooms": [2, 3, 5],
            "bathrooms": [1, 2, 3],
            "age": [10, 5, 1],
            "location_score": [5, 7, 9],
        })
        scaled = scaler.transform(sample)
        preds = model.predict(scaled)
        assert preds.shape == (3,), f"Expected 3 predictions, got {preds.shape}"
        assert all(p > 0 for p in preds), "All predictions should be positive"

    def test_feature_count(self, artifacts):
        model, scaler = artifacts
        sample = pd.DataFrame([{
            "sqft": 2000, "bedrooms": 3, "bathrooms": 2,
            "age": 5, "location_score": 7,
        }])
        assert sample.shape[1] == 5, f"Expected 5 features, got {sample.shape[1]}"
        scaled = scaler.transform(sample)
        assert scaled.shape[1] == 5, f"Scaled output should have 5 features, got {scaled.shape[1]}"

    def test_feature_importances(self, artifacts):
        model, _ = artifacts
        importances = model.feature_importances_
        assert len(importances) == 5, f"Expected 5 feature importances, got {len(importances)}"
        assert np.isclose(sum(importances), 1.0), "Feature importances should sum to ~1.0"
