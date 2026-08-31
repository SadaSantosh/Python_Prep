import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("🚀 DAY 12 (ENTERPRISE EDITION): Preprocessing & Domain Feature Engineering...")

raw_data_path = os.path.join(BASE_DIR, "telco_churn.csv")
df = pd.read_csv(raw_data_path)

print(f"📊 Initial Raw Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
print(f"✅ Imputed NaNs in TotalCharges with Median: ${total_charges_median:.2f}")

tenure_safe = df['tenure'].replace(0, 1)
df['Avg_Monthly_Cost'] = df['TotalCharges'] / tenure_safe
df['Monthly_Price_Diff'] = df['MonthlyCharges'] - df['Avg_Monthly_Cost']
print("💡 Engineered 2 New Signals: 'Avg_Monthly_Cost' & 'Monthly_Price_Diff'")

if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
    print("🗑️ Dropped non-predictive column: 'customerID'")

binary_map_cols = ['Churn', 'PaperlessBilling', 'Partner', 'Dependents']
for col in binary_map_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
print(f"🔤 One-Hot Encoding {len(categorical_cols)} Categorical Features...")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Cost', 'Monthly_Price_Diff']
scaler = StandardScaler()
df_encoded[num_cols_to_scale] = scaler.fit_transform(df_encoded[num_cols_to_scale])
print("⚙️ Applied StandardScaler to numerical features (including engineered ratios)")

scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
joblib.dump(scaler, scaler_path)
print(f"💾 Saved production scaler to '{scaler_path}'")

null_count = df_encoded.isnull().sum().sum()
assert null_count == 0, f"⚠️ Pipeline Error: Found {null_count} null values!"
print("🛡️ Quality Audit Passed: 0 Null values detected.")

clean_data_path = os.path.join(BASE_DIR, "telco_churn_cleaned.csv")
df_encoded.to_csv(clean_data_path, index=False)
print(f"\n✅ Pipeline Complete! Output Matrix Shape: {df_encoded.shape[0]} Rows, {df_encoded.shape[1]} Columns")
print(f"💾 Processed data saved to '{clean_data_path}'!")
