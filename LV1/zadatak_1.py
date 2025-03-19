#Zadatak #1

#Napišite program koji od korisnika zahtijeva unos radnih sati te koliko je plaćen
#po radnom satu. Koristite ugrađenu Python metodu input(). Nakon toga izračunajte koliko
#je korisnik zaradio i ispišite na ekran. Na kraju prepravite rješenje na način da ukupni iznos
#izračunavate u zasebnoj funkciji naziva total_euro.

#Primjer:
#Radni sati: 35 h
#eura/h: 8.5
#Ukupno: 297.5 eura

def total_euro(hours, hour_pay):
    return hours * hour_pay

work_hours = float(input("Work hours: "))
pay_by_the_hour = float(input("euros/h: "))

total = total_euro(work_hours, pay_by_the_hour)

print(f"Total: {total} euros") 
