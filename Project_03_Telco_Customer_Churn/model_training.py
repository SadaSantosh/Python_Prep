import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("🚀 DAY 13: Handling Class Imbalance with SMOTE & Training Churn Models...")

clean_data_path = os.path.join(BASE_DIR, "telco_churn_cleaned.csv")
df = pd.read_csv(clean_data_path)

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Original Training Class Distribution:\n{y_train.value_counts()}")

smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\n⚖️ Resampled Training Class Distribution via SMOTE:\n{y_train_resampled.value_counts()}")

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_resampled, y_train_resampled)
y_pred_lr = lr_model.predict(X_test)

print("\n--------------------------------------------------")
print("📈 Model 1: Logistic Regression (SMOTE-Balanced)")
print("--------------------------------------------------")
print(f"🎯 Test Accuracy: {accuracy_score(y_test, y_pred_lr) * 100:.2f}%")
print(f"🎯 Churn Recall (Sensitivity): {recall_score(y_test, y_pred_lr) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

rf_model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
rf_model.fit(X_train_resampled, y_train_resampled)
y_pred_rf = rf_model.predict(X_test)

print("\n--------------------------------------------------")
print("🌲 Model 2: Random Forest Classifier (SMOTE-Balanced)")
print("--------------------------------------------------")
print(f"🎯 Test Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
print(f"🎯 Churn Recall (Sensitivity): {recall_score(y_test, y_pred_rf) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

xgb_model = XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)
xgb_model.fit(X_train_resampled, y_train_resampled)
y_pred_xgb = xgb_model.predict(X_test)

print("\n--------------------------------------------------")
print("⚡ Model 3: XGBoost Classifier (SMOTE-Balanced)")
print("--------------------------------------------------")
print(f"🎯 Test Accuracy: {accuracy_score(y_test, y_pred_xgb) * 100:.2f}%")
print(f"🎯 Churn Recall (Sensitivity): {recall_score(y_test, y_pred_xgb) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

models = {
    "Logistic Regression": (lr_model, recall_score(y_test, y_pred_lr)),
    "Random Forest": (rf_model, recall_score(y_test, y_pred_rf)),
    "XGBoost": (xgb_model, recall_score(y_test, y_pred_xgb)),
}
best_name, (best_model, best_recall) = max(models.items(), key=lambda item: item[1][1])

model_path = os.path.join(BASE_DIR, "best_churn_model.pkl")
joblib.dump(best_model, model_path)
print(f"\n💾 Exported best model ({best_name}, recall={best_recall * 100:.2f}%) to '{model_path}'!")
