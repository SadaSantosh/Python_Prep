"""Preprocess the raw employee dataset: dedupe, impute, and one-hot encode."""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_PATH = os.path.join(BASE_DIR, "employee_data.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "employee_data_cleaned.csv")

print("=" * 60)
print("EMPLOYEE DATA PREPROCESSING PIPELINE")
print("=" * 60)

# 1. Load the raw dataset
df = pd.read_csv(RAW_PATH)
print(f"\nInitial shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Drop exact duplicate rows
duplicate_count = df.duplicated().sum()
print(f"Exact duplicate rows found: {duplicate_count}")
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows after deduplication: {df.shape[0]}")

# 3. Inspect and impute missing values
print("\n--- Missing Value Inspection ---")
print(df.isnull().sum())

df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Years_Experience"] = df["Years_Experience"].fillna(df["Years_Experience"].median())
print("Missing continuous features imputed (Age: mean, Years_Experience: median).")

# 4. One-hot encode categorical columns so models receive numeric inputs only
print("\nEncoding categorical attributes...")
attrition_col = df["Attrition"] if "Attrition" in df.columns else None
performance_col = df["Performance_Score"] if "Performance_Score" in df.columns else None

df_cleaned = pd.get_dummies(df, columns=["Department", "Remote_Worker"], dtype=int)

if performance_col is not None:
    perf_dummies = pd.get_dummies(performance_col, prefix="Performance", dtype=int)
    df_cleaned = pd.concat(
        [df_cleaned.drop(columns=["Performance_Score"], errors="ignore"), perf_dummies],
        axis=1,
    )
if attrition_col is not None:
    df_cleaned["Attrition"] = attrition_col.values

print(f"Final preprocessed shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")
df_cleaned.to_csv(CLEAN_PATH, index=False)
print(f"Cleaned dataset saved to '{CLEAN_PATH}'.")
