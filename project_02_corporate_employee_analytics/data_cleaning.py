import pandas as pd

print("=========================================")
print("   ADVANCED DATA PREPROCESSING SYSTEM    ")
print("=========================================")

# 1. Load the real-world dataset matrix
df = pd.read_csv("employee_data.csv")
print(f"📊 Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Identify and drop duplicate rows
duplicate_count = df.duplicated().sum()
print(f"🔍 Found {duplicate_count} exact duplicate rows.")
df = df.drop_duplicates().reset_index(drop=True)
print(f"🧼 Data after removing duplicates: {df.shape[0]} rows")

# 3. Check for missing values (NaN entries)
print("\n--- Missing Value Inspection ---")
print(df.isnull().sum())

# Impute continuous missing columns using the column mean values
df["Age"] = df["Age"].fillna(df["Age"].mean())
df["Years_Experience"] = df["Years_Experience"].fillna(df["Years_Experience"].median())
print("✅ Missing continuous features imputed successfully.")

# 4. Advanced Transformation: One-Hot Encoding Text Columns
# ML models cannot read raw strings like "IT", "HR", "Sales". We convert them to 0s and 1s.
print("\n🔄 Converting categorical attributes to numerical matrices...")
attrition_col = df["Attrition"] if "Attrition" in df.columns else None
performance_col = df["Performance_Score"] if "Performance_Score" in df.columns else None
df_cleaned = pd.get_dummies(df, columns=["Department", "Remote_Worker"], dtype=int)
if performance_col is not None:
    perf_dummies = pd.get_dummies(performance_col, prefix="Performance", dtype=int)
    df_cleaned = pd.concat([df_cleaned.drop(columns=["Performance_Score"], errors="ignore"), perf_dummies], axis=1)
if attrition_col is not None:
    df_cleaned["Attrition"] = attrition_col.values

print("\n--- PROCESSED DATAFRAME PREVIEW ---")
print(df_cleaned.head())
print(f"\n🎯 Final Preprocessed Shape: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns")

# Save cleaned production-ready state to disk
df_cleaned.to_csv("employee_data_cleaned.csv", index=False)
print("💾 Cleaned dataset saved as 'employee_data_cleaned.csv'!")