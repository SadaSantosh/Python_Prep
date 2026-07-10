import pandas as pd 

print("________Day 4: Loading External CSV Data________")

# Load the csv file into a DataFrame
df = pd.read_csv("student_data.csv")
df["Hours_Studied"] = df["Hours_Studied"].fillna(0) # Fill missing values in the "Hours_Studied" column with 0

#Printing the entire table 
print(df)