"""Visualize logistic regression and decision tree boundaries on student data.

The scatter plot and both decision surfaces are drawn on the two most
interpretable features (Hours_Studied x Sleep_Hours). Boundary models are fit
on that 2D plane separately from the full 3-feature models used for the
accuracy comparison, so each visual is mathematically honest.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "student_data.csv")

FEATURE_COLS = ["Hours_Studied", "Sleep_Hours", "Attendance"]
PLANE_COLS = ["Hours_Studied", "Sleep_Hours"]


def main():
    df = pd.read_csv(DATA_PATH)
    df["Hours_Studied"] = df["Hours_Studied"].fillna(df["Hours_Studied"].mean())

    x = df[FEATURE_COLS]
    y = df["Passed"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Full models used for the accuracy comparison
    lr_model = LogisticRegression()
    lr_model.fit(x_train, y_train)

    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(x_train, y_train)

    # Scatter of actual students colored by outcome
    sns.scatterplot(data=df, x=PLANE_COLS[0], y=PLANE_COLS[1], hue="Passed", style="Passed", s=100)

    # 2D logistic regression boundary
    plane_x = df[PLANE_COLS]
    lr_plane = LogisticRegression()
    lr_plane.fit(plane_x, y)

    x_vals = np.linspace(plane_x["Hours_Studied"].min(), plane_x["Hours_Studied"].max(), 100)
    y_vals = -(lr_plane.intercept_[0] + lr_plane.coef_[0][0] * x_vals) / lr_plane.coef_[0][1]
    plt.plot(x_vals, y_vals, color="black", linestyle="--", label="Logistic Boundary")
    plt.legend()

    # 2D decision tree boundary rendered as a shaded region
    tree_plane = DecisionTreeClassifier(max_depth=3)
    tree_plane.fit(plane_x, y)

    x_min = plane_x["Hours_Studied"].min() - 1
    x_max = plane_x["Hours_Studied"].max() + 1
    y_min = plane_x["Sleep_Hours"].min() - 1
    y_max = plane_x["Sleep_Hours"].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1),
    )

    z = tree_plane.predict(np.c_[xx.ravel(), yy.ravel()])
    z_num = np.where(z == "Yes", 1, 0)
    plt.contourf(xx, yy, z_num.reshape(xx.shape), alpha=0.2, cmap="coolwarm")

    # Report accuracy of the full 3-feature models on the held-out set
    lr_preds = lr_model.predict(x_test)
    tree_preds = tree_model.predict(x_test)
    print(f"Logistic Regression accuracy: {accuracy_score(y_test, lr_preds):.2%}")
    print(f"Decision Tree accuracy:       {accuracy_score(y_test, tree_preds):.2%}")

    plt.title("Student Performance: Study Hours vs Sleep Hours")
    plt.xlabel("Hours Studied")
    plt.ylabel("Hours of Sleep")
    plt.show()


if __name__ == "__main__":
    main()
