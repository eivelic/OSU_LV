#Zadatak #2

#Napišite program koji od korisnika zahtijeva upis jednog broja koji predstavlja
#nekakvu ocjenu i nalazi se između 0.0 i 1.0. Ispišite kojoj kategoriji pripada ocjena na temelju
#sljedećih uvjeta:

# >= 0.9 A
# >= 0.8 B
# >= 0.7 C
# >= 0.6 D
# < 0.6 F

def grade_category(score):
    if score >= 0.9:
        return "A"
    elif score >= 0.8:
        return "B"
    elif score >= 0.7:
        return "C"
    elif score >= 0.6:
        return "D"
    else:
        return "F"

try:
    user_input = input("Enter a grade (between 0.0 and 1.0): ")
    score = float(user_input)

    if 0.0 <= score <= 1.0:
        print(f"Category: {grade_category(score)}")
    else:
        print("Error: The entered number is out of range (0.0 to 1.0).")

except ValueError:
    print("Error: Please enter a valid decimal number.")
