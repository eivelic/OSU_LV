#Napišite programski kod koji će prikazati sljedeće vizualizacije:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("data_C02_emission.csv")

#a) Pomoću histograma prikažite emisiju C02 plinova. Komentirajte dobiveni prikaz.
data['CO2 Emissions (g/km)'].plot(kind = 'hist', color = 'green', edgecolor = 'white')
plt.title("CO2 Emissions")
plt.show()

#b) Pomoću dijagrama raspršenja prikažite odnos između gradske potrošnje goriva i emisije C02 plinova.
#Komentirajte dobiveni prikaz. Kako biste bolje razumjeli odnose između veličina, obojite točkice na dijagramu
#raspršenja s obzirom na tip goriva.
data['Fuel Type'] = data['Fuel Type'].astype("category")
data.plot.scatter(
    x='Fuel Consumption City (L/100km)',
    y='CO2 Emissions (g/km)',
    c='Fuel Type',
    cmap="twilight"
)
plt.title("CO2 Emissions/Fuel Consumption City")
plt.show()

#Komentar:
#Odnos između gradske potrošnje goriva i emisije CO2 plinova je poprilično ravnopravan, tj. linearan te proporcionalan.
#S druge strane, iako se Ethanol (E85) također linearno razvija, kroz istu potrošnju goriva emitira manje ugljikovog dioksida
#u odnosu na primjerice dizel i benzin.

#c) Pomoću kutijastog dijagrama prikažite razdiobu izvangradske potrošnje s obzirom na tip goriva. 
#Primjećujete li grubu mjernu pogrešku u podacima?
data.boxplot(column=["Fuel Consumption Hwy (L/100km)"], by="Fuel Type")
plt.title("Distribution of Highway Fuel Consumption by Fuel Type")
plt.suptitle('')
plt.show()

#d) Pomoću stupčastog dijagrama prikažite broj vozila po tipu goriva. Koristite metodu groupby.
fuel_type = data.groupby('Fuel Type').size()

fuel_type.plot(kind='bar', color='skyblue', edgecolor='white')
plt.title("Number of Vehicles by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Number of Vehicles")
plt.show()

#e) Pomoću stupčastog grafa prikažite na istoj slici prosječnu C02 emisiju vozila s obzirom na broj cilindara.
average_co2_emission = data.groupby('Cylinders')['CO2 Emissions (g/km)'].mean()
average_co2_emission.plot(kind = 'bar', color='skyblue', edgecolor='white')
plt.title("Cylinders/Avg CO2 Emission")
plt.ylabel("Average CO2 Emission")
plt.show()
