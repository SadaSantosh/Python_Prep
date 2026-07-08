import pandas as pd 
import matplotlib.pyplot as plt

print("____________________ Day 3:Data Visualization Intro __________________")

# 1. workout progression dataset 
workout_data = {
    "week": [1, 2, 3, 4, 5, 6, 7],
    "Max_Pull_Ups": [4, 5, 6, 7, 8, 9, 11]
}

# 2. Convert to DataFrame
df = pd.DataFrame(workout_data)

# 3. Create a clean line plot graph to visualize the progression of pull-ups over the weeks
plt.bar(df["week"], df["Max_Pull_Ups"], color='crimson') # bar graph with crimson color

# 4. Add professional labels and title to the graph
plt.title("Calisthenics Progression: Max Pull-Ups Over Time")
plt.xlabel("Traning Weeks")
plt.ylabel("Repetitions Achieved")
plt.grid(True) #Adds a clean backdroud grid 

# 5. Display the graph window onto your screen 
print("\nLunching interactive graph interface . . .")
plt.show() # this command will open a new window with the graph and allow you to interact with it