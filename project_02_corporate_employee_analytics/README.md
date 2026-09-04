# 🏢 Corporate Employee Analytics

Workforce attrition analysis pipeline: synthetic data generation, cleaning, EDA, and a Random Forest attrition predictor.

## 📦 Pipeline Order

1. **`generate_data.py`** — Creates a realistic synthetic employee dataset (1,215 rows incl. intentional duplicates and missing values).
2. **`data_cleaning.py`** — Deduplicates, imputes missing values, and one-hot encodes categorical columns.
3. **`data_analysis.py`** — Winsorizes salary outliers, computes the correlation matrix, saves a heatmap, and reports attrition by department.
4. **`model_training.py`** — Trains a class-balanced Random Forest on attrition and exports `random_forest_model.pkl`.

## 🚀 Running

```bash
pip install -r requirements.txt

python generate_data.py     # creates employee_data.csv
python data_cleaning.py     # creates employee_data_cleaned.csv
python data_analysis.py     # creates correlation_heatmap.png
python model_training.py    # creates random_forest_model.pkl
```

## 👤 Author

Engineered by **Sada Santosh Kalmath**
