import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the expanded dataset
df = pd.read_csv("student_data.csv")

# Fill missing values if any
df["Hours_Studied"] = df["Hours_Studied"].fillna(df["Hours_Studied"].mean())

# 2. Train with all THREE features now!
x = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
y = df["Passed"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train models
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(x_train, y_train)

# Print the real accuracies with the new feature
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_model.predict(x_test))}")
print(f"Decision Tree Accuracy: {accuracy_score(y_test, tree_model.predict(x_test))}")

print("\n--- Making a Custom Prediction ---")
# 3. Predict for a custom student scenario
# Let's test: 4.5 Hours Studied, 7 Hours Sleep, 85% Attendance
custom_student = pd.DataFrame([[4.5, 7.0, 85]], columns=["Hours_Studied", "Sleep_Hours", "Attendance"])

lr_pred = lr_model.predict(custom_student)
tree_pred = tree_model.predict(custom_student)

print(f"Logistic Regression predicts: {lr_pred[0]}")
print(f"Decision Tree predicts: {tree_pred[0]}")