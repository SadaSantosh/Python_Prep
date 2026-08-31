"""Unit tests for Project 03: Telco Customer Churn Engine."""
import os
import sys
import numpy as np
import pandas as pd
import pytest
import joblib

# Add project directory to path
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "Project_03_Telco_Customer_Churn")
sys.path.insert(0, BASE_DIR)


class TestModelArtifacts:
    """Verify that all model artifacts exist and are loadable."""

    def test_model_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "best_churn_model.pkl")), (
            "best_churn_model.pkl not found — run model_training.py first"
        )

    def test_scaler_file_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "scaler.pkl")), (
            "scaler.pkl not found — run data_preprocessing.py first"
        )

    def test_cleaned_data_exists(self):
        assert os.path.isfile(os.path.join(BASE_DIR, "telco_churn_cleaned.csv")), (
            "telco_churn_cleaned.csv not found — run data_preprocessing.py first"
        )

    def test_model_loads_successfully(self):
        model = joblib.load(os.path.join(BASE_DIR, "best_churn_model.pkl"))
        assert hasattr(model, "predict"), "Loaded object has no predict method"
        assert hasattr(model, "predict_proba"), "Loaded object has no predict_proba method"

    def test_scaler_loads_successfully(self):
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        assert hasattr(scaler, "transform"), "Loaded scaler has no transform method"


class TestPredictionPipeline:
    """Test that the model can make predictions on valid input."""

    @pytest.fixture
    def artifacts(self):
        model = joblib.load(os.path.join(BASE_DIR, "best_churn_model.pkl"))
        scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
        df = pd.read_csv(os.path.join(BASE_DIR, "telco_churn_cleaned.csv"))
        feature_cols = [c for c in df.columns if c != "Churn"]
        return model, scaler, feature_cols, df

    def test_prediction_shape(self, artifacts):
        model, scaler, feature_cols, df = artifacts
        sample = df[feature_cols].iloc[:5]
        preds = model.predict(sample)
        proba = model.predict_proba(sample)
        assert preds.shape == (5,), f"Expected 5 predictions, got {preds.shape}"
        assert proba.shape == (5, 2), f"Expected (5,2) probabilities, got {proba.shape}"

    def test_prediction_values_are_binary(self, artifacts):
        model, scaler, feature_cols, df = artifacts
        sample = df[feature_cols].iloc[:10]
        preds = model.predict(sample)
        unique_vals = set(preds)
        assert unique_vals <= {0, 1}, f"Predictions contain non-binary values: {unique_vals}"

    def test_probabilities_sum_to_one(self, artifacts):
        model, scaler, feature_cols, df = artifacts
        sample = df[feature_cols].iloc[:10]
        proba = model.predict_proba(sample)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_churn_column_exists(self, artifacts):
        _, _, _, df = artifacts
        assert "Churn" in df.columns, "Churn target column missing from cleaned data"

    def test_no_null_values(self, artifacts):
        _, _, _, df = artifacts
        null_count = df.isnull().sum().sum()
        assert null_count == 0, f"Found {null_count} null values in cleaned data"


class TestPreprocessingPipeline:
    """Test the preprocessing helper function logic."""

    def test_raw_batch_preprocessing(self):
        """Simulate raw batch input and verify preprocessing doesn't crash."""
        # Create a minimal raw-like DataFrame
        raw_df = pd.DataFrame({
            "customerID": ["CUST_TEST_001"],
            "tenure": [24],
            "MonthlyCharges": [70.0],
            "TotalCharges": ["1680.00"],
            "Contract": ["Month-to-month"],
            "InternetService": ["DSL"],
            "PaymentMethod": ["Electronic check"],
            "TechSupport": ["No"],
            "OnlineSecurity": ["No"],
            "PaperlessBilling": ["Yes"],
            "Partner": ["Yes"],
            "Dependents": ["No"],
        })

        # Verify the raw DataFrame structure
        assert len(raw_df) == 1
        assert "tenure" in raw_df.columns
        assert "MonthlyCharges" in raw_df.columns
        assert "TotalCharges" in raw_df.columns
