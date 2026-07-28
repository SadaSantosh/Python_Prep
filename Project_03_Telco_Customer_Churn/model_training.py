import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib

print("🚀 DAY 13: Handling Class Imbalance with SMOTE & Training Churn Models...")

# 1. Load the preprocessed 33-column dataset
clean_data_path = "project_03_telco_customer_churn/telco_churn_cleaned.csv"
df = pd.read_csv(clean_data_path)

# 2. Separate Features (X) and Target (y)
X = df.drop(columns=['Churn'])
y = df['Churn']

# 3. Train/Test Split (Stratified to maintain baseline ratios in test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Original Training Class Distribution:\n{y_train.value_counts()}")

# 4. Apply SMOTE ONLY on Training Data (Prevents Data Leakage into Test Set)
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"\n⚖️ Resampled Training Class Distribution via SMOTE:\n{y_train_resampled.value_counts()}")

# 5. Model 1: Logistic Regression (Baseline Linear Classifier)
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

# 6. Model 2: Random Forest Classifier (Non-Linear Ensemble)
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

# 7. Export Best Model (Random Forest) for Web App Deployment
joblib.dump(rf_model, "project_03_telco_customer_churn/best_churn_model.pkl")
print("\n💾 Exported best model binary to 'project_03_telco_customer_churn/best_churn_model.pkl'!")