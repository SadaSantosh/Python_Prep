import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 1. Load the expanded dataset
df = pd.read_csv("student_data.csv")

# Fill any missing values just like before
df["Hours_Studied"] = df["Hours_Studied"].fillna(df["Hours_Studied"].mean())

# Training the model on the data 
x = df[["Hours_Studied", "Sleep_Hours", "Attendance"]]
y = df["Passed"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

tree_model = DecisionTreeClassifier(max_depth=3)
tree_model.fit(x_train, y_train)

# 2. Create the scatter plot
sns.scatterplot(data=df, x="Hours_Studied", y="Sleep_Hours", hue="Passed", style="Passed", s=100)

# 3. Calculate and plot the Decision Boundary line
import numpy as np 

# Getting the weights from our traning model 
intercept = model.intercept_[0]
coef_hours, coef_sleep = model.coef_[0]

# Creating X values spanning across our graph (from min hours to max hours studied)
x_values = np.linspace(df["Hours_Studied"].min(), df["Hours_Studied"].max(), 100)

# Calculating the matching Y values (Sleep_Hours) using the logistic regression boundary formula
y_values = -(intercept + coef_hours * x_values) / coef_sleep

# Plotting the boundary line 
plt.plot(x_values, y_values, color='black', linestyle='--', label='Decision Boundary')
plt.legend()

# 4. Visualize the Decision Tree Boundary using a colored grid
import numpy as np

# Create a mesh grid of points covering the entire graph area
x_min, x_max = df["Hours_Studied"].min() - 1, df["Hours_Studied"].max() + 1
y_min, y_max = df["Sleep_Hours"].min() - 1, df["Sleep_Hours"].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# Predict the outcome for every single point on the grid using the Tree model
Z = tree_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Convert string predictions ('Yes'/'No') to numbers for background shading
Z_num = np.where(Z == 'Yes', 1, 0)

# Draw the colored background blocks (Alpha makes it transparent so we see the points)
plt.contourf(xx, yy, Z_num, alpha=0.2, cmap='coolwarm')

# Generating predictions on the test set
lr_preds = model.predict(x_test)
tree_preds = tree_model.predict(x_test)

# Printing the accuracy scores 
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_preds))
print("Decision Tree Accuracy:", accuracy_score(y_test, tree_preds))

plt.title("Student Performance: Study Hours vs Sleep Hours")
plt.xlabel("Hours Studied")
plt.ylabel("Hours of Sleep")
plt.show()
