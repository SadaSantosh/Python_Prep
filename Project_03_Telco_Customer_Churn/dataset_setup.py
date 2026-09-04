import pandas as pd
import os

print("🚀 DAY 11: Loading Telco Customer Churn Real-World Dataset...")

# Real-world Telco Churn CSV Source URL
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

# Load directly into Pandas
df = pd.read_csv(url)

# Define local save path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(BASE_DIR, "telco_churn.csv")
df.to_csv(save_path, index=False)

print(f"✅ Dataset successfully downloaded and saved to '{save_path}'!")
print(f"📊 Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns\n")

# Preview first 5 rows
print("🔍 First 5 Rows Preview:")
print(df.head())
