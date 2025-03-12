#Datoteka data.csv sadrži mjerenja visine i mase provedena na muškarcima i
#ženama. Skripta zadatak_2.py učitava dane podatke u obliku numpy polja data pri čemu je u
#prvom stupcu polja oznaka spola (1 muško, 0 žensko), drugi stupac polja je visina u cm, a treći
#stupac polja je masa u kg.

import numpy as np
import matplotlib.pyplot as plt

#a) Na temelju veličine numpy polja data, na koliko osoba su izvršena mjerenja?

data = np.loadtxt('data.csv', delimiter=',', skiprows = 1)  

num_people = data.shape[0]
print("Zadatak #1:")
print(f"Broj osoba na kojima su izvršena mjerenja: {num_people}")

#b) Prikažite odnos visine i mase osobe pomoću naredbe matplotlib.pyplot.scatter.

heights = data[:, 1] 
weights = data[:, 2] 

plt.scatter(heights, weights, alpha=0.5)
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Odnos visine i mase")
plt.show()

#c) Ponovite prethodni zadatak, ali prikažite mjerenja za svaku pedesetu osobu na slici.

plt.scatter(data[::50,1], data[::50,2], alpha=0.5, color="lightblue")
plt.xlabel("Visina (cm)")
plt.ylabel("Masa (kg)")
plt.title("Dijagram ovisnosti visine i mase za svaku 50. osobu")
plt.show()

#d) Izračunajte i ispišite u terminal minimalnu, maksimalnu i srednju vrijednost visine u ovom
#podatkovnom skupu.

heights = data[:, 1]

min_height = np.min(heights)
max_height = np.max(heights)
mean_height = np.mean(heights)

print("Zadatak #4:")
print(f"Minimalna visina: {min_height:.2f} cm")
print(f"Maksimalna visina: {max_height:.2f} cm")
print(f"Srednja visina: {mean_height:.2f} cm")

#e) Ponovite zadatak pod d), ali samo za muškarce, odnosno žene. Npr. kako biste izdvojili
#muškarce, stvorite polje koje zadrži bool vrijednosti i njega koristite kao indeks retka.
#ind = (data[:,0] == 1)

male_indices = (data[:, 0] == 1)
female_indices = (data[:, 0] == 0)

male_heights = data[male_indices, 1]
female_heights = data[female_indices, 1]

min_male_height = np.min(male_heights)
max_male_height = np.max(male_heights)
mean_male_height = np.mean(male_heights)

print("Zadatak #5:")
print(f"Muškarci - Minimalna visina: {min_male_height:.2f} cm")
print(f"Muškarci - Maksimalna visina: {max_male_height:.2f} cm")
print(f"Muškarci - Srednja visina: {mean_male_height:.2f} cm")

min_female_height = np.min(female_heights)
max_female_height = np.max(female_heights)
mean_female_height = np.mean(female_heights)

print(f"Žene - Minimalna visina: {min_female_height:.2f} cm")
print(f"Žene - Maksimalna visina: {max_female_height:.2f} cm")
print(f"Žene - Srednja visina: {mean_female_height:.2f} cm")
