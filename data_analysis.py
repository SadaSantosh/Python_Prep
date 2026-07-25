import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

print("=========================================")
print("   DAY 9: OUTLIER & CORRELATION SYSTEM   ")
print("=========================================")

# 1. Loading yesterday's cleaned dataset 
df = pd.read_csv("employee_data_cleaned.csv")
print(f"📊 Loaded Cleaned Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns ")

# 2. Outlier Detection using IQR ( Interquartile Range )
print("\n 🔍 Checking for Statistical Outliers in 'Salary'...")
Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)
IQR = Q3 - Q1 
lower_bound = Q1 - 1.5 * IQR 
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df["Salary"] < lower_bound) | (df["Salary"] > upper_bound)]
print(f" Outliers Detected in Salary: {len(outliers)}")

#Capping outliers to boundary limits (Winsorization)
df["Salary"] = np.where(df["Salary"] > upper_bound, upper_bound, 
                        np.where(df["Salary"] < lower_bound, lower_bound, df["Salary"]))
print("✅ Outliers capped successfully within normal distrubution bounds.")

#3. Exclude non-numeric ID column for correlation
numeric_df = df.drop(columns=["Employee_ID", "Performance_Score"])

#4. Compute Correlation Matrix
print("\n 📈 Computing Feature Correlation Matrix....")
corr_matrix = numeric_df.corr()
print(corr_matrix["Age"].sort_values(ascending=False))

#5. Render Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5 )
plt.title("Day 9: Employee Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
print("\n Heatmap saved as 'correlation_heatmap.png'!!!!!")
plt.show()