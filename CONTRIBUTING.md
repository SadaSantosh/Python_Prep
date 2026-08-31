# 🤝 Contributing to Python_Prep

Thank you for your interest in contributing! This guide will help you get set up and understand the project structure.

---

## 📁 Project Structure

```
Python_Prep/
├── project_01_student_performance_classifier/   # ML: Student pass/fail prediction
│   ├── ml_basic.py              # Pandas intro & data filtering
│   ├── ml_viz.py                # Logistic regression + decision tree visualization
│   ├── ml_predict.py            # Interactive CLI prediction system
│   ├── student_data.csv         # Training dataset
│   ├── requirements.txt         # Project-specific dependencies
│   └── *.pkl                    # Saved model artifacts
│
├── project_02_corporate_employee_analytics/      # EDA: Employee attrition analysis
│   ├── data_analysis.py         # Outlier detection, correlation heatmap
│   ├── model_training.py        # Random Forest attrition predictor
│   ├── employee_data.csv        # Raw employee dataset
│   ├── employee_data_cleaned.csv
│   └── requirements.txt         # Project-specific dependencies
│
├── Project_03_Telco_Customer_Churn/              # Streamlit: Churn prediction engine
│   ├── app.py                   # Streamlit frontend (Minimalist UI)
│   ├── data_preprocessing.py    # Feature engineering & scaling
│   ├── model_training.py        # SMOTE + XGBoost/RF/LR training
│   ├── advanced_eval.py         # Model comparison & feature importance
│   ├── eda_audit.py             # Dataset imbalance audit
│   ├── dataset_setup.py         # Download raw telco data
│   ├── requirements.txt         # Project-specific dependencies
│   ├── .streamlit/config.toml   # Streamlit theme config
│   └── *.pkl, *.csv             # Model artifacts & data
│
├── project_04_real_estate_ai/                    # Streamlit: Property valuation
│   ├── app.py                   # Streamlit frontend (Minimalist UI)
│   ├── train_model.py           # Random Forest regression training
│   ├── requirements.txt         # Project-specific dependencies
│   ├── .streamlit/config.toml   # Streamlit theme config
│   └── *.pkl, *.json            # Model artifacts & metrics
│
├── project_05_spam_phishing_detector/            # Streamlit: Spam/phishing detection
│   ├── app.py                   # Streamlit frontend (Minimalist UI)
│   ├── train_model.py           # TF-IDF + Naive Bayes training
│   ├── requirements.txt         # Project-specific dependencies
│   ├── .streamlit/config.toml   # Streamlit theme config
│   └── *.pkl, *.json            # Model artifacts & metrics
│
├── tests/                                       # Unit tests
│   ├── conftest.py              # Shared pytest fixtures & sys.path setup
│   ├── test_project03_churn.py  # 11 tests: model loading, predictions, preprocessing
│   ├── test_project04_realestate.py  # 10 tests: artifacts, predictions, features
│   └── test_project05_spam.py   # 14 tests: NLP pipeline, URL heuristics
│
├── .github/workflows/ci.yml    # GitHub Actions CI/CD pipeline
├── requirements.txt             # Global dependencies
├── README.md                    # Portfolio overview
└── CONTRIBUTING.md              # This file
```

---

## 🚀 Local Development Setup

### Prerequisites
- Python 3.10+
- Git

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/SadaSantosh/Python_Prep.git
cd Python_Prep
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
# Global dependencies (covers all projects)
pip install -r requirements.txt

# For running tests
pip install pytest
```

### 3. Run Individual Projects

```bash
# Project 01 — Student classifier (CLI)
cd project_01_student_performance_classifier
python ml_predict.py

# Project 02 — Employee analytics (script)
cd project_02_corporate_employee_analytics
python data_analysis.py

# Project 03 — Telco churn (Streamlit app)
cd Project_03_Telco_Customer_Churn
streamlit run app.py

# Project 04 — Real estate AI (Streamlit app)
cd project_04_real_estate_ai
streamlit run app.py

# Project 05 — Spam detector (Streamlit app)
cd project_05_spam_phishing_detector
streamlit run app.py
```

### 4. Retrain Models (Optional)

```bash
# Project 03
cd Project_03_Telco_Customer_Churn
python dataset_setup.py      # Downloads telco data
python data_preprocessing.py  # Feature engineering
python model_training.py      # Train SMOTE-balanced models

# Project 04
cd project_04_real_estate_ai
python train_model.py

# Project 05
cd project_05_spam_phishing_detector
python train_model.py
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run tests for a specific project
pytest tests/test_project03_churn.py -v
pytest tests/test_project04_realestate.py -v
pytest tests/test_project05_spam.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term-missing
```

### Test Coverage Summary

| Test File | Tests | What's Covered |
|-----------|-------|----------------|
| `test_project03_churn.py` | 11 | Model/scaler loading, prediction shape & values, probability sums, null checks, preprocessing |
| `test_project04_realestate.py` | 10 | Artifact loading, R² validation, single/batch predictions, feature importances |
| `test_project05_spam.py` | 14 | Model/TF-IDF loading, ham/spam classification, probabilities, URL TLD/IP/length detection |

---

## 🎨 UI Design — Minimalist

All Streamlit apps (Projects 03, 04, 05) use a **Minimalist** UI with:
- Clean white sidebar with subtle border
- White metric cards with soft shadows
- Dark buttons with clean rounded corners
- Subtle background image overlay
- Inter font from Google Fonts

Theme configs are in `.streamlit/config.toml` for each Streamlit project.

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push/PR to `main`:

1. **Test Job** — Python 3.10/3.11/3.12 matrix, syntax validation of all `.py` files, pytest execution
2. **Lint Job** — flake8 static analysis (non-blocking)
3. **Deploy-readiness Job** — Validates Streamlit apps pass tests

### Streamlit Cloud Deployment

Each Streamlit app is deployed separately on [Streamlit Community Cloud](https://streamlit.io/cloud):
- **Telco Churn**: `Project_03_Telco_Customer_Churn/app.py`
- **Real Estate AI**: `project_04_real_estate_ai/app.py`
- **Spam Detector**: `project_05_spam_phishing_detector/app.py`

---

## 📝 Code Style

- **Line length**: 120 characters max
- **Formatting**: Follow existing patterns in each project
- **Imports**: Group by standard lib → third-party → local
- **Type hints**: Encouraged for function signatures
- **Docstrings**: Use `"""triple quotes"""` for classes and functions
- **Naming**: `snake_case` for variables/functions, `PascalCase` for classes

---

## 🐛 Bug Reports

When reporting a bug, please include:
1. Which project/file is affected
2. Steps to reproduce
3. Expected vs actual behavior
4. Python version and OS

---

## 📄 License

This project is for educational purposes. Built by **Sada Santosh Kalmath**.
