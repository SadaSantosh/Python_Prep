import random 
# Setting up my fighters using variables
fighter_namae = "Gojo Satoru"
fighter_hp = 100

fighter2_name = "Ryomen Sukuna"
fighter2_hp = 100

print(f"Domain Expansion: {fighter_namae} vs {fighter2_name}!")

# The loop runs as long as BOTH fighters have HP above 0
while fighter_hp > 0 and fighter2_hp > 0:
    
    #Gojo Attacks!
    attack_dmage = random.randint(15,30)
    fighter2_hp = fighter2_hp - attack_dmage 
    print(f"{fighter_namae} uses Hollow Purple and deals {attack_dmage} damage!! Sukuna HP: {fighter2_hp}")

    #checks if Sukuna is defeated 
    if fighter2_hp < 0:
        print(f"\n**** {fighter2_name} is defeated!! {fighter_namae} WINS!!!!!!! ***")
        break #this command breakes us out of the loop adn ends theh game 

    Sukuna_damage = random.randint(15,30)
    fighter_hp = fighter_hp - Sukuna_damage
    print(f"{fighter2_name} uses Cleave and deals {Sukuna_damage} damage! Gojo HP: {fighter_hp}\n")

    if fighter_hp <= 0:
        print(f"\n*** {fighter_namae} is defeated! {fighter2_name} Wins! ***")
        break 