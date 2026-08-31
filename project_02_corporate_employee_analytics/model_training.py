import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("🚀 DAY 10: Training Advanced Ensemble Pipeline...")

# 1. Load the cleaned dataset directly from your workspace
df = pd.read_csv("employee_data_cleaned.csv")

# 2. Use Attrition as the primary HR prediction target
if "Attrition" not in df.columns:
    raise ValueError("Attrition column missing. Re-run generate_data.py and data_cleaning.py first.")

target_col = "Attrition"
print(f"🎯 Target column: '{target_col}'")

# 3. Separate Features (X) and Target Label (y)
# Drop both the target column AND non-numeric ID columns
cols_to_drop = [target_col]
if "Employee_ID" in df.columns:
    cols_to_drop.append("Employee_ID")

X = df.drop(columns=cols_to_drop)  # Predictors (numbers only)
y = df[target_col]                 # Target Class

# 4. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5. Initialize & Train Random Forest Ensemble Model
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42
)
rf_model.fit(X_train, y_train)

# 6. Model Inference & Evaluation
y_pred = rf_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Training Complete!")
print(f"🎯 Test Accuracy: {acc * 100:.2f}%\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# 7. Feature Importance Extraction
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("💡 Top 3 Predictive Features:")
print(importances.head(3))

# 8. Pipeline Export
joblib.dump(rf_model, "random_forest_model.pkl")
print("\n💾 Saved production pipeline as 'random_forest_model.pkl'!")