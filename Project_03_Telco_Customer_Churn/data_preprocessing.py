import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

print("🚀 DAY 12 (ENTERPRISE EDITION): Preprocessing & Domain Feature Engineering...")

# 1. Load the raw Telco dataset
raw_data_path = "project_03_telco_customer_churn/telco_churn.csv"
df = pd.read_csv(raw_data_path)

print(f"📊 Initial Raw Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")

# 2. Fix 'TotalCharges' string whitespace & coerce to float
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])

# 3. Impute missing values using column median
total_charges_median = df['TotalCharges'].median()
df['TotalCharges'] = df['TotalCharges'].fillna(total_charges_median)
print(f"✅ Imputed NaNs in TotalCharges with Median: ${total_charges_median:.2f}")

# 4. 🧠 DOMAIN FEATURE ENGINEERING (Derived Business Signals)
# Feature 1: Historical Average Monthly Charge vs Current Monthly Charge
# Avoid division by zero for brand new customers (tenure = 0)
tenure_safe = df['tenure'].replace(0, 1)
df['Avg_Monthly_Cost'] = df['TotalCharges'] / tenure_safe

# Feature 2: Monthly Price Spike Indicator
df['Monthly_Price_Diff'] = df['MonthlyCharges'] - df['Avg_Monthly_Cost']

print("💡 Engineered 2 New Signals: 'Avg_Monthly_Cost' & 'Monthly_Price_Diff'")

# 5. Drop non-predictive primary key
if 'customerID' in df.columns:
    df = df.drop(columns=['customerID'])
    print("🗑️ Dropped non-predictive column: 'customerID'")

# 6. Map Binary Target ('Churn') and Yes/No text features to 1/0
binary_map_cols = ['Churn', 'PaperlessBilling', 'Partner', 'Dependents']
for col in binary_map_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

# 7. One-Hot Encode remaining categorical string features
categorical_cols = df.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
print(f"🔤 One-Hot Encoding {len(categorical_cols)} Categorical Features...")

df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Convert boolean dummies to 1/0 integers
bool_cols = df_encoded.select_dtypes(include=['bool']).columns
df_encoded[bool_cols] = df_encoded[bool_cols].astype(int)

# 8. Apply Feature Scaling to continuous numerical predictors
num_cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Avg_Monthly_Cost', 'Monthly_Price_Diff']
scaler = StandardScaler()

df_encoded[num_cols_to_scale] = scaler.fit_transform(df_encoded[num_cols_to_scale])
print(f"⚙️ Applied StandardScaler to numerical features (including engineered ratios)")

# 9. Save Scaler artifact for future Streamlit web app inference
joblib.dump(scaler, "project_03_telco_customer_churn/scaler.pkl")
print("💾 Saved production scaler to 'project_03_telco_customer_churn/scaler.pkl'")

# 10. Quality Assertion
null_count = df_encoded.isnull().sum().sum()
assert null_count == 0, f"⚠️ Pipeline Error: Found {null_count} null values!"
print("🛡️ Quality Audit Passed: 0 Null values detected.")

# 11. Export processed matrix
clean_data_path = "project_03_telco_customer_churn/telco_churn_cleaned.csv"
df_encoded.to_csv(clean_data_path, index=False)
print(f"\n✅ Pipeline Complete! Output Matrix Shape: {df_encoded.shape[0]} Rows, {df_encoded.shape[1]} Columns")
print(f"💾 Processed data saved to '{clean_data_path}'!")