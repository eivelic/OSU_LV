#Skripta zadatak_1.py učitava podatkovni skup iz data_C02_emission.csv.
#Dodajte programski kod u skriptu pomoću kojeg možete odgovoriti na sljedeća pitanja:
import pandas as pd

data = pd.read_csv('data_C02_emission.csv')

#a) Koliko mjerenja sadrži DataFrame?
print(len(data))

#Kojeg je tipa svaka veličina?
print(data.info())

#Postoje li izostale ili duplicirane vrijednosti?
#Obrišite ih ako postoje.
data.drop_duplicates()

#Kategoričke veličine konvertirajte u tip category
#data['Make'] = data['Make'].astype('category')
#data['Model'] = data['Model'].astype('category')
#data['Vehicle Class'] = data['Vehicle Class'].astype('category')
#data['Transmission'] = data['Transmission'].astype('category')
#data['Fuel Type'] = data['Fuel Type'].astype('category')

#print(data.info())

#b) Koja tri automobila ima najveću odnosno najmanju gradsku potrošnju? Ispišite u terminal:
#ime proizvođača, model vozila i kolika je gradska potrošnja.
most_fuel_consumption = data.sort_values('Fuel Consumption City (L/100km)', ascending=False).head(3)
print("Top three cars with most fuel consumption:")
for index, row in most_fuel_consumption.iterrows():
    print(f"{row['Make']} {row['Model']}: {row['Fuel Consumption City (L/100km)']} L/100km")

least_fuel_consumption = data.sort_values('Fuel Consumption City (L/100km)').head(3)
print("\nTop three cars with least fuel consumption:")
for index, row in least_fuel_consumption.iterrows():
    print(f"{row['Make']} {row['Model']}: {row['Fuel Consumption City (L/100km)']} L/100km")

#c) Koliko vozila ima veličinu motora između 2.5 i 3.5 L?
engines = data[(data['Engine Size (L)'] > 2.5) & (data['Engine Size (L)'] < 3.5)]
print(len(engines))

#Kolika je prosječna C02 emisija plinova za ova vozila?
average_co2_emission = data['CO2 Emissions (g/km)'].mean()
print(average_co2_emission)

#d) Koliko mjerenja se odnosi na vozila proizvođača Audi?
audi_makers = data[(data['Make'] == 'Audi')]
print(len(audi_makers))

#Kolika je prosječna emisija C02 plinova automobila proizvođača Audi koji imaju 4 cilindara?
audi_4_cylinders = audi_makers[audi_makers['Cylinders'] == 4]
average_audi_co2_emission = audi_4_cylinders['CO2 Emissions (g/km)'].mean()
print(average_audi_co2_emission)

#e) Koliko je vozila s 4,6,8... cilindara?
even_cylinder_vehicles = data[(data['Cylinders'] % 2 == 0) & (data['Cylinders'] >= 4)]
even_cylinder_vehicles_count = len(even_cylinder_vehicles)
print(even_cylinder_vehicles_count)

#Kolika je prosječna emisija C02 plinova s obzirom na broj cilindara?
avg_co2_emission = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
print(avg_co2_emission)

#f) Kolika je prosječna gradska potrošnja u slučaju vozila koja koriste dizel, a kolika za vozila
#koja koriste regularni benzin?
diesel_vehicles = data[data['Fuel Type'] == 'D']
gasoline_vehicles = data[data['Fuel Type'] == 'X']

average_city_consumption_diesel = diesel_vehicles['Fuel Consumption City (L/100km)'].mean()
average_city_consumption_gasoline = gasoline_vehicles['Fuel Consumption City (L/100km)'].mean()

print(f"Average fuel city consumption for diesel vehicles: {average_city_consumption_diesel}")
print(f"Average fuel city consumption for gasoline vehicles: {average_city_consumption_gasoline}")

#Koliko iznose medijalne vrijednosti?
diesel_median = diesel_vehicles['Fuel Consumption City (L/100km)'].median()
gasoline_median = gasoline_vehicles['Fuel Consumption City (L/100km)'].median()

print(f"Median values for diesel vehicles: {diesel_median}")
print(f"Median values for gasoline vehicles: {gasoline_median}")

#g) Koje vozilo s 4 cilindra koje koristi dizelski motor ima najveću gradsku potrošnju goriva?
four_cylinder_vehicle = data[(data['Cylinders'] == 4) & (data['Fuel Type'] == 'D')]
most_fuel_consumed_vehicle = four_cylinder_vehicle.sort_values(by = "Fuel Consumption City (L/100km)").head(1)
print(f"Vozilo s 4 cilindra koje koristi dizelski motor te ima najveću gradsku potrošnju goriva: {most_fuel_consumed_vehicle}")

#h) Koliko ima vozila ima ručni tip mjenjača (bez obzira na broj brzina)?
manual_transmission_vehicles = data[data['Transmission'].str.startswith('M')]
print(len(manual_transmission_vehicles))

#i)Izračunajte korelaciju između numeričkih veličina. Komentirajte dobiveni rezultat.
correlation = data.corr(numeric_only = True)
print(correlation)

#Komentar:
#Prikazani rezultat je zapravo matrica koja prikazuje korelaciju numeričkih veličina u rasponu od -1 do 1.
#Primjerice, očekivano, veličina motora i broj cilindara su snažno povezani – veći motori obično imaju više cilindara i to predstavlja jako pozitivnu korelaciju jer je bliže 1.
#S druge strane, negativna korelacija bi bio odnos vozila s višim mpg i njihova emisija CO2. Takva vozila emitiraju manje te odnos teži -1.
