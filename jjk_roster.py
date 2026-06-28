# Creating a list of characters 
roster = ["Gojo Satoru", " Ryomen Sukuna", "Yuji Itadori", "Megumi Fushiguro"]
print("___________ WELCOME TO THE JUJUTSU ROSTER ___________")
print(f"\nAvailable fighters:{roster}")

#Getting choice from the player using input()
#The computer will pause here and wait for the player to type his choosen character in terminal!
player_choice = input("\nType the name of theh fighter you choose: ")
#Checking if the player has typed OR choosen character is inour list
if player_choice in roster:
    print(f"Excellent choice!! You have locked in {player_choice}.")
else:
    print("That character is not in the roster!! Did u check your spelling??")

print("\n ____________ List Manipulation Practice ___________")
print("\n A new character approches . . .")
roster.append("Nobara Kugisaki")

print(f"Updated Roster: {roster}")
print(f"The total number of fighters is now: {len(roster)}")
