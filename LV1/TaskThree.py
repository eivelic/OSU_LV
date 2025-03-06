#Zadatak #3

#Napišite program koji od korisnika zahtijeva unos brojeva u beskonačnoj petlji
#sve dok korisnik ne upiše "Done" (bez navodnika). Pri tome brojeve spremajte u listu. Nakon toga
#potrebno je ispisati koliko brojeva je korisnik unio, njihovu srednju, minimalnu i maksimalnu
#vrijednost. Sortirajte listu i ispišite je na ekran. Dodatno: osigurajte program od pogrešnog unosa
#(npr. slovo umjesto brojke) na način da program zanemari taj unos i ispiše odgovarajuću poruku.

def main():
    numbers = []

    while True:
        user_input = input("Enter a number (or 'Done' to finish): ")

        if user_input == "Done":  
            break 

        try:
            number = int(user_input)  
            numbers.append(number)  
        except ValueError:
            print("Invalid input! Please enter a valid number.")  

    if numbers:
        print(f"\nTotal numbers entered: {len(numbers)}")
        print(f"Average: {sum(numbers) / len(numbers):.2f}")
        print(f"Minimum: {min(numbers)}")
        print(f"Maximum: {max(numbers)}")
        numbers.sort()
        print(f"Sorted list: {numbers}")
    else:
        print("No valid numbers were entered.")

main()
