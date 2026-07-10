import pandas as pd
from sklearn.linear_model import LogisticRegression

print("________Day 4: Training my First AI Model________")

# 1. Loading and cleaning the dataset
df = pd.read_csv("student_data.csv")
df["Hours_Studied"] = df["Hours_Studied"].fillna(0) 

# 2. splitting data into Input Features (X) and Output Target (y)
x = df[["Hours_Studied"]] # Input must be a @D grid/DataFrame
y = df["Passed"]          # Target is a 1D series/column

print("Features (X):")
print(x)
print("\nTarget (y):")
print(y)

# 3. Initializing and Training the machine learning model
model = LogisticRegression()
model.fit(x, y) # This is wheer the AI learns the patterns!!

# 4. Using the trained model to make a new Predicition 
new_student_hours = pd.DataFrame([[1]], columns=["Hours_Studied"]) # Lets predict for student who studied for 5 hours
prediction = model.predict(new_student_hours)

print(f"\nPrediction for student who studied 5 hours: {prediction[0]}")
