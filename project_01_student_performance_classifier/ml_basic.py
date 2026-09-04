"""Intro to Pandas: structuring and filtering a small student dataset."""

import pandas as pd

print("=" * 60)
print("PANDAS DATA ENGINEERING: STUDENT PERFORMANCE DATASET")
print("=" * 60)

# 1. Define a small real-world-style dataset
raw_data = {
    "Student_Name": ["Sada", "Santosh", "Yuji", "Megumi"],
    "Student_Hours_Per_Day": [6, 7, 2, 4],
    "Passed_Exam": ["Yes", "Yes", "No", "No"],
}

# 2. Convert the raw rows into a structured DataFrame (rows and columns)
df = pd.DataFrame(raw_data)
print("\nStructured dataset:")
print(df)

# 3. Basic analytics: average study hours across all students
average_hours = df["Student_Hours_Per_Day"].mean()
print(f"\nAverage hours studied per day: {average_hours:.2f}")

print("\n" + "=" * 60)
print("FILTERING DATA IN PANDAS")
print("=" * 60)

# 4. Conditional filtering: students who passed the exam
passed_students = df[df["Passed_Exam"] == "Yes"]
print("\nStudents who passed the exam:")
print(passed_students)
