import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# File paths for saving our trained models
LR_MODEL_PATH = "logistic_regression_model.pkl"
TREE_MODEL_PATH = "decision_tree_model.pkl"

# Check if the models are already trained and saved
if not os.path.exists(LR_MODEL_PATH) or not os.path.exists(TREE_MODEL_PATH):
    print("🤖 First-time setup: Training models and saving them to disk...")
    df = pd.read_csv("student_data.csv")
    df["Hours_Studied"] = df["Hours_Studied"].fillna(df["Hours_Studied"].mean())

    x = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
    y = df["Passed"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)

    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(x_train, y_train)

    # Save models using pickle
    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr_model, f)
    with open(TREE_MODEL_PATH, "wb") as f:
        pickle.dump(tree_model, f)
    print("💾 Models successfully saved as .pkl files!\n")
else:
    print("⚡ Fast load: Loading pre-trained models from disk...")
    # Load models using pickle
    with open(LR_MODEL_PATH, "rb") as f:
        lr_model = pickle.load(f)
    with open(TREE_MODEL_PATH, "rb") as f:
        tree_model = pickle.load(f)

# Interactive Looping Prediction System
print("=========================================")
print("   STUDENT PASS/FAIL PREDICTION SYSTEM   ")
print("=========================================")

while True:
    print("\n[1] Predict a Student's Outcome")
    print("[2] Exit Program")
    choice = input("Select an option (1 or 2): ").strip()

    if choice == '2':
        print("\n👋 Exiting the system. Have a great day!")
        break
    elif choice == '1':
        print("\n--- Enter Student Data ---")
        try:
            hours = float(input("Enter Hours Studied (0 to 10): "))
            sleep = float(input("Enter Sleep Hours (0 to 10): "))
            attendance = float(input("Enter Attendance Percentage (0 to 100): "))

            custom_student = pd.DataFrame([[hours, sleep, attendance]], 
                                           columns=["Hours_Studied", "Sleep_Hours", "Attendance"])

            lr_pred = lr_model.predict(custom_student)[0]
            tree_pred = tree_model.predict(custom_student)[0]

            print("\n================ RESULTS ================")
            print(f"📐 Logistic Regression Predicts: {lr_pred}")
            print(f"🧱 Decision Tree Classifier Predicts: {tree_pred}")
            print("=========================================")

        except ValueError:
            print("\n❌ Error: Please enter valid numbers!")
    else:
        print("\n❌ Invalid choice! Please type 1 or 2.")