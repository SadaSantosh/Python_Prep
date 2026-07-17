import pandas as pd
import numpy as np

# Set seed for reproducibility
np.random.seed(42)
n_rows = 1200

# Continuous choices and probabilities verified to match perfectly
age_choices = [22, 25, 30, 35, 40, 45, 50, np.nan]
age_probs   = [0.1, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1, 0.05]

exp_choices = [1, 2, 3, 5, 7, 10, 15, np.nan]
exp_probs   = [0.15, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05, 0.05]

data = {
    "Employee_ID": [f"EMP_{i:04d}" for i in range(1, n_rows + 1)],
    "Age": np.random.choice(age_choices, size=n_rows, p=age_probs),
    "Department": np.random.choice(["IT", "HR", "Sales", "Marketing"], size=n_rows),
    "Years_Experience": np.random.choice(exp_choices, size=n_rows, p=exp_probs),
    "Salary": np.random.randint(40000, 120000, size=n_rows),
    "Remote_Worker": np.random.choice(["Yes", "No"], size=n_rows),
    "Performance_Score": np.random.choice(["Low", "Medium", "High"], size=n_rows)
}

# Duplicate some rows intentionally to make it messy
df = pd.DataFrame(data)
df = pd.concat([df, df.iloc[:15]], ignore_index=True)

# Save to file system
df.to_csv("employee_data.csv", index=False)
print("📦 employee_data.csv generated successfully with 1,215 records!")