"""Exploratory analysis: salary outlier capping and feature correlation."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CLEAN_PATH = os.path.join(BASE_DIR, "employee_data_cleaned.csv")
RAW_PATH = os.path.join(BASE_DIR, "employee_data.csv")
HEATMAP_PATH = os.path.join(BASE_DIR, "correlation_heatmap.png")

print("=" * 60)
print("OUTLIER & CORRELATION ANALYSIS")
print("=" * 60)

# 1. Load the cleaned dataset
df = pd.read_csv(CLEAN_PATH)
print(f"Loaded cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Detect salary outliers with the IQR rule, then cap them (Winsorization)
print("\nChecking for statistical outliers in 'Salary'...")
q1 = df["Salary"].quantile(0.25)
q3 = df["Salary"].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]
print(f"Outliers detected in Salary: {len(outliers)}")

df["Salary"] = np.where(
    df["Salary"] > upper_bound,
    upper_bound,
    np.where(df["Salary"] < lower_bound, lower_bound, df["Salary"]),
)
print("Outliers capped to the boundary limits (Winsorization).")

# 3. Build a numeric-only frame for correlation analysis
drop_cols = ["Employee_ID", "Performance_Score", "Attrition"]
if "Attrition" in df.columns:
    df["Attrition_Binary"] = (df["Attrition"] == "Yes").astype(int)
numeric_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# 4. Compute and inspect the correlation matrix
print("\nComputing feature correlation matrix...")
corr_matrix = numeric_df.corr()
print(corr_matrix["Age"].sort_values(ascending=False))

# 5. Render and persist the correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Employee Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(HEATMAP_PATH, dpi=300)
print(f"Heatmap saved as '{HEATMAP_PATH}'.")

# 6. Attrition rate by department (from the raw data before encoding)
raw_df = pd.read_csv(RAW_PATH)
if "Attrition" in raw_df.columns and "Department" in raw_df.columns:
    print("\nAttrition rate by department:")
    dept_attrition = (
        raw_df.groupby("Department")["Attrition"]
        .apply(lambda x: (x == "Yes").mean() * 100)
        .sort_values(ascending=False)
    )
    for dept, rate in dept_attrition.items():
        print(f"  {dept}: {rate:.1f}%")

plt.show()
