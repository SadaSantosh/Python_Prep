"""Generate the synthetic employee attrition dataset used by the HR analytics pipeline."""

import os

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, "employee_data.csv")

# Deterministic seed keeps regenerated datasets comparable
np.random.seed(42)
N_ROWS = 1200

# Age and experience draws include NaNs to give the cleaning step realistic gaps
age_choices = [22, 25, 30, 35, 40, 45, 50, np.nan]
age_probs = [0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]

exp_choices = [1, 2, 3, 5, 7, 10, 15, np.nan]
exp_probs = [0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]

data = {
    "Employee_ID": [f"EMP_{i:04d}" for i in range(1, N_ROWS + 1)],
    "Age": np.random.choice(age_choices, size=N_ROWS, p=age_probs),
    "Department": np.random.choice(["IT", "HR", "Sales", "Marketing"], size=N_ROWS),
    "Years_Experience": np.random.choice(exp_choices, size=N_ROWS, p=exp_probs),
    "Salary": np.random.randint(40000, 120000, size=N_ROWS),
    "Remote_Worker": np.random.choice(["Yes", "No"], size=N_ROWS),
    "Performance_Score": np.random.choice(["Low", "Medium", "High"], size=N_ROWS),
}

df = pd.DataFrame(data)

# Derive attrition from realistic workforce signals
attrition_prob = (
    0.10
    + (df["Remote_Worker"] == "No").astype(float) * 0.06
    + (df["Salary"] < 50000).astype(float) * 0.14
    + (df["Years_Experience"].fillna(0) < 2).astype(float) * 0.08
)
df["Attrition"] = np.where(np.random.rand(N_ROWS) < attrition_prob, "Yes", "No")

# Duplicate a handful of rows to give the cleaning step duplicates to remove
df = pd.concat([df, df.iloc[:15]], ignore_index=True)

df.to_csv(OUTPUT_PATH, index=False)
print(f"employee_data.csv generated successfully with {len(df)} records.")
