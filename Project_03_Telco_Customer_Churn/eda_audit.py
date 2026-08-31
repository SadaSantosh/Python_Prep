import pandas as pd

print("🔍 DAY 11: Auditing Telco Dataset for Imbalance & Data Types...")

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, "telco_churn.csv"))

# 1. Target Class Distribution Analysis
churn_counts = df['Churn'].value_counts()
churn_percentages = df['Churn'].value_counts(normalize=True) * 100

print("\n🎯 Target Variable ('Churn') Distribution:")
for category in churn_counts.index:
    print(f"   • {category}: {churn_counts[category]} customers ({churn_percentages[category]:.2f}%)")

print("\n💡 Key Insight: Notice the class imbalance! ~73% stayed vs ~26% churned.")
print("   Standard models trained on this will bias toward 'No Churn'. We will solve this with SMOTE later!\n")

# 2. Check for hidden empty strings in 'TotalCharges'
# In real datasets, numerical columns often contain spaces ' ' instead of nulls!
empty_spaces = (df['TotalCharges'] == ' ').sum()
print(f"⚠️ Found {empty_spaces} rows with blank space ' ' strings in TotalCharges!")

# 3. Print Data Types
print("\n📋 Column Data Types Overview:")
print(df.dtypes.head(10))