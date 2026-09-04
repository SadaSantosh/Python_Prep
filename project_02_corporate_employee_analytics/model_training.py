"""Train a Random Forest attrition classifier and export it for reuse."""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, "employee_data_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "random_forest_model.pkl")

print("=" * 60)
print("EMPLOYEE ATTRITION MODEL TRAINING")
print("=" * 60)

# 1. Load the cleaned dataset
df = pd.read_csv(CLEAN_PATH)

# 2. Use attrition as the prediction target
if "Attrition" not in df.columns:
    raise ValueError("Attrition column missing. Re-run generate_data.py and data_cleaning.py first.")

target_col = "Attrition"
print(f"Target column: '{target_col}'")

# 3. Separate features from the target; drop identifiers and leftover text columns
cols_to_drop = [target_col]
if "Employee_ID" in df.columns:
    cols_to_drop.append("Employee_ID")

for col in df.drop(columns=cols_to_drop).columns:
    if df[col].dtype == "object":
        cols_to_drop.append(col)

x = df.drop(columns=cols_to_drop)
y = df[target_col]

# 4. Train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 5. Train the ensemble with class weights to counter class imbalance
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
)
rf_model.fit(x_train, y_train)

# 6. Evaluate on the held-out test set
y_pred = rf_model.predict(x_test)
acc = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {acc * 100:.2f}%\n")
print("Classification report:")
print(classification_report(y_test, y_pred))

# 7. Report the strongest predictive features
importances = pd.Series(rf_model.feature_importances_, index=x.columns).sort_values(ascending=False)
print("\nTop 3 predictive features:")
print(importances.head(3))

# 8. Export the trained model
joblib.dump(rf_model, MODEL_PATH)
print(f"Saved production pipeline as '{MODEL_PATH}'.")
