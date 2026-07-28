import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

print("🚀 DAY 14: Training XGBoost & Generating Feature Importance Analysis...")

# 1. Load Data
df = pd.read_csv("project_03_telco_customer_churn/telco_churn_cleaned.csv")
X = df.drop(columns=['Churn'])
y = df['Churn']

# 2. Train/Test Split & SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# 3. Initialize & Train XGBoost
xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_res, y_train_res)

# 4. Train Baselines for Unified Comparison
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_res, y_train_res)
rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42).fit(X_train_res, y_train_res)

models = {
    "Logistic Regression": lr,
    "Random Forest": rf,
    "XGBoost Classifier": xgb_model
}

# 5. Build Model Performance Leaderboard Matrix
results = []
for name, model in models.items():
    preds = model.predict(X_test)
    results.append({
        "Model": name,
        "Accuracy": f"{accuracy_score(y_test, preds)*100:.2f}%",
        "Recall (Churn)": f"{recall_score(y_test, preds)*100:.2f}%",
        "Precision": f"{precision_score(y_test, preds)*100:.2f}%",
        "F1-Score": f"{f1_score(y_test, preds)*100:.2f}%"
    })

comparison_df = pd.DataFrame(results)
print("\n🏆 MODEL LEADERBOARD COMPARISON MATRIX:")
print(comparison_df.to_string(index=False))

# 6. Extract Top 10 Feature Importances from XGBoost
importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title("🔥 Top 10 Churn Drivers (XGBoost Feature Importance)")
plt.xlabel("Relative Importance Score")
plt.ylabel("Feature")
plt.tight_layout()

chart_path = "project_03_telco_customer_churn/xgb_feature_importance.png"
plt.savefig(chart_path, dpi=300)
print(f"\n📊 Feature Importance Chart saved to '{chart_path}'!")

# 7. Save XGBoost as Final Best Production Pipeline
joblib.dump(xgb_model, "project_03_telco_customer_churn/best_churn_model.pkl")
print("💾 Overwrote 'best_churn_model.pkl' with production XGBoost model!")