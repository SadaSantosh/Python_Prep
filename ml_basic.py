import pandas as pd

print("____________________ AI/ML Data Engineering Intro __________________")

# 1. Simulation a real-world dataset
raw_data = {
    "Student_Name": ["Sada", "Santosh", "Yuji", "Megumi"],
    "Student_Hours_Per_Day": [6,7,2,4], 
    "Passed_Exam": ["Yes", "Yes", "No", "No"]
}

#2. Converting raw data into a structured Pandas DataFram (Rows & Columns)
df = pd.DataFrame(raw_data)
print("\nOur Sturctured Dataset:")
print(df)

#3. Basic Data Analytics: Calculating the average hours spent by students per day
average_hours = df["Student_Hours_Per_Day"].mean()
print(f"\nThe average hours spent by students per day is: {average_hours}")
print("\n____________________Filtering Data in Pandas______________________")

#4. Filtering data based on a condition: Students who passed the exam
passed_students = df[df["Passed_Exam"] == "Yes"]
print("\nStudents who passed the exam:")
print(passed_students)