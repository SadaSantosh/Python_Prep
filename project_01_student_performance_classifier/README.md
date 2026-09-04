# 🎓 Student Performance Classifier

Classification pipeline that predicts whether a student passes the exam from study hours, sleep hours, and attendance.

## 📦 What's Inside

- **`ml_basic.py`** — Pandas intro: structuring raw rows into a DataFrame and filtering.
- **`ml_viz.py`** — Visualizes logistic regression and decision tree decision boundaries (2D plane) with model accuracy comparison.
- **`ml_predict.py`** — Interactive CLI that trains and persists both classifiers, then serves pass/fail predictions with probabilities.

## 🚀 Running

```bash
pip install -r requirements.txt

# Explore the dataset with Pandas
python ml_basic.py

# See the model decision boundaries (opens a matplotlib window)
python ml_viz.py

# Train (on first run) and start the interactive predictor
python ml_predict.py
```

Model artifacts (`logistic_regression_model.pkl`, `decision_tree_model.pkl`) are reused when present; delete them to force retraining.

## 👤 Author

Engineered by **Sada Santosh Kalmath**
