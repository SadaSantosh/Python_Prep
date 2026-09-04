"""Interactive CLI that predicts a student's pass/fail outcome.

Trains logistic regression and decision tree classifiers on first run,
persists both models to disk, then reuses them for fast inference.
"""

import os
import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")
LR_MODEL_PATH = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
TREE_MODEL_PATH = os.path.join(BASE_DIR, "decision_tree_model.pkl")

FEATURE_COLS = ["Hours_Studied", "Sleep_Hours", "Attendance"]


def load_or_train_models():
    """Return trained (logistic_regression, decision_tree) classifiers.

    Pre-trained models are reused when present; otherwise both models are
    trained, validated, and persisted for future sessions.
    """
    if os.path.exists(LR_MODEL_PATH) and os.path.exists(TREE_MODEL_PATH):
        with open(LR_MODEL_PATH, "rb") as f:
            lr_model = pickle.load(f)
        with open(TREE_MODEL_PATH, "rb") as f:
            tree_model = pickle.load(f)
        print("Loaded pre-trained models from disk.")
        return lr_model, tree_model

    df = pd.read_csv(DATA_PATH)
    df["Hours_Studied"] = df["Hours_Studied"].fillna(df["Hours_Studied"].mean())

    x = df[FEATURE_COLS]
    y = df["Passed"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)

    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(x_train, y_train)

    lr_acc = accuracy_score(y_test, lr_model.predict(x_test))
    tree_acc = accuracy_score(y_test, tree_model.predict(x_test))
    print(f"Validation accuracy -> Logistic Regression: {lr_acc:.2%} | Decision Tree: {tree_acc:.2%}")

    with open(LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr_model, f)
    with open(TREE_MODEL_PATH, "wb") as f:
        pickle.dump(tree_model, f)
    print("Models trained and saved to disk.")
    return lr_model, tree_model


def predict_student(lr_model, tree_model):
    """Prompt for student metrics and print both model predictions."""
    print("\n--- Enter Student Data ---")
    try:
        hours = float(input("Enter Hours Studied (0 to 10): "))
        sleep = float(input("Enter Sleep Hours (0 to 10): "))
        attendance = float(input("Enter Attendance Percentage (0 to 100): "))
    except ValueError:
        print("\nError: Please enter valid numbers.")
        return

    if not (0 <= hours <= 10 and 0 <= sleep <= 10 and 0 <= attendance <= 100):
        print("\nError: Values must be within valid ranges (Hours/Sleep: 0-10, Attendance: 0-100).")
        return

    student = pd.DataFrame([[hours, sleep, attendance]], columns=FEATURE_COLS)

    lr_pred = lr_model.predict(student)[0]
    tree_pred = tree_model.predict(student)[0]
    lr_pass_idx = list(lr_model.classes_).index("Yes")
    tree_pass_idx = list(tree_model.classes_).index("Yes")
    lr_prob = lr_model.predict_proba(student)[0][lr_pass_idx] * 100
    tree_prob = tree_model.predict_proba(student)[0][tree_pass_idx] * 100

    print("\n================ RESULTS ================")
    print(f"Logistic Regression predicts: {lr_pred} (Pass probability: {lr_prob:.1f}%)")
    print(f"Decision Tree predicts:       {tree_pred} (Pass probability: {tree_prob:.1f}%)")
    print("=========================================")


def main():
    lr_model, tree_model = load_or_train_models()

    print("=========================================")
    print("   STUDENT PASS/FAIL PREDICTION SYSTEM   ")
    print("=========================================")

    while True:
        print("\n[1] Predict a Student's Outcome")
        print("[2] Exit Program")
        choice = input("Select an option (1 or 2): ").strip()

        if choice == "2":
            print("\nExiting the system. Have a great day!")
            break
        if choice == "1":
            predict_student(lr_model, tree_model)
        else:
            print("\nInvalid choice! Please type 1 or 2.")


if __name__ == "__main__":
    main()
