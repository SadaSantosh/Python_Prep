import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("________Day 4: Training my First AI Model________")

# 1. Loading and cleaning the dataset
df = pd.read_csv("student_data.csv")
df = df.fillna(0)  

# 2. splitting data into Input Features (X) and Output Target (y)
x = df[["Hours_Studied", "Sleep_Hours"]] # Input must be a @D grid/DataFrame
y = df["Passed"]          # Target is a 1D series/column

#splitting the data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 3. Initializing and Training the machine learning model
model = LogisticRegression()
model.fit(x_train, y_train) # This is wheer the AI learns the patterns!!

# ---- NEW: Model Evalution ----
from sklearn.metrics import accuracy_score

# Generate predicitions for our traning data 
predictions = model.predict(x_test)

# Calculate the accuracy percentage 
accuracy = accuracy_score(y_test, predictions)
print(f"\nTest Model Accuracy: {accuracy * 100}%")
print(f"Model Predictions: {predictions}")

# 4. Using the trained model to make a new Prediction 
new_student_hours = pd.DataFrame([[0.0, 0.0]], columns=["Hours_Studied", "Sleep_Hours"]) # Lets predict for student who studied for 5 hours
prediction = model.predict(new_student_hours)

print(f"\nPrediction for student who studied 1 hour and slept 8 hours: {prediction[0]}")
