import pandas as pd 
import matplotlib.pyplot as plt

print("____________________ Day 3:Data Visualization Intro __________________")

# 1. workout progression dataset 
workout_data = {
    "week": [1, 2, 3, 4, 5, 6, 7],
    "Max_Pull_Ups": [4, 5, 6, 7, 8, 9, 11],
    "Yuji_Max_Pull_Ups": [3, 3, 5, 5, 8, 8, 10] #new comarison column added to the dataset
}

# 2. Convert to DataFrame
df = pd.DataFrame(workout_data)

# 3. Create a clean line plot graph to visualize the progression of pull-ups over the weeks
plt.plot(df["week"], df["Max_Pull_Ups"], marker='o', color='crimson', label='Your Progress ', linewidth=2) # bar graph with crimson color
plt.plot(df["week"], df["Yuji_Max_Pull_Ups"], marker='s', color='royalblue', label='Yuji Progress', linewidth=2)
plt.legend() # adds a legend to the graph to identify the two lines

# 4. Add professional labels and title to the graph
plt.title("Calisthenics Progression: Max Pull-Ups Over Time")
plt.xlabel("Traning Weeks")
plt.ylabel("Repetitions Achieved")
plt.grid(True) #Adds a clean backdroud grid 

# 5. Display the graph window onto your screen 
print("\nLunching interactive graph interface . . .")

# 6. Traning Time Distribution Dataset 
routine_data = {
    "Activity" : ["Strength Training", "cardio", "Flexibility", "Reset/Recovery" ],
    "Hours" : [5, 3, 2, 2]
}

# Convert to new DataFrame
df_routine = pd.DataFrame(routine_data)

# Creating a clean Pie Chart 
plt.figure() # opens a fresh blank canvas for the new graph
plt.pie(df_routine["Hours"], labels=df_routine["Activity"], autopct='%1.1f%%', colors={'crimson', 'royalblue', 'gold', 'seagreen'})
plt.title("Weekly Training Time Distribution")
plt.savefig("training_time_distribution.png", dpi=300) #Saves the graph as a PNG file in the current working directory 
plt.show() # this command will open a new window with the graph and allow you to interact with it