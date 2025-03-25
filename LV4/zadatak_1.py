# Skripta zadatak_1.py učitava podatkovni skup iz data_C02_emission.csv.
# Potrebno je izgraditi i vrednovati model koji procjenjuje emisiju C02 plinova na temelju ostalih
# numeričkih ulaznih veličina. Detalje oko ovog podatkovnog skupa mogu se pronaći u 3.
# laboratorijskoj vježbi.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_C02_emission.csv')

# a) Odaberite željene numeričke veličine specificiranjem liste s nazivima stupaca.
# Podijelite podatke na skup za učenje i skup za testiranje u omjeru 80%-20%.

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
print("Numerički stupci:", numeric_columns)

X = df[numeric_columns]
y = df['CO2 Emissions (g/km)']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(f"Skup za učenje X_train: {X_train.shape}")
print(f"Skup za testiranje X_test: {X_test.shape}")

# b) Pomoću matplotlib biblioteke i dijagrama raspršenja prikažite ovisnost emisije C02 plinova
# o jednoj numeričkoj veličini. Pri tome podatke koji pripadaju skupu za učenje označite
# plavom bojom, a podatke koji pripadaju skupu za testiranje označite crvenom bojom.

x_train_selected = X_train['Engine Size (L)']
x_test_selected = X_test['Engine Size (L)']

plt.figure(figsize=(10, 6))

plt.scatter(x_train_selected, y_train, color='blue')
plt.scatter(x_test_selected, y_test, color='red')

plt.xlabel('Veličina motora (L)')
plt.ylabel('Emisija CO2 (g/km)')
plt.title('Ovisnost emisije CO2 o veličini motora')
plt.show()

# c) Izvršite standardizaciju ulaznih veličina skupa za učenje. Prikažite histogram vrijednosti
# jedne ulazne veličine prije i nakon skaliranja. Na temelju dobivenih parametara skaliranja
# transformirajte ulazne veličine skupa podataka za testiranje.

from sklearn.preprocessing import StandardScaler

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(x_train_selected, bins=20, color='royalblue', edgecolor='black')
plt.xlabel('Engine Size (L)')
plt.ylabel('Frekvencija')
plt.title('Histogram prije skaliranja')

sc = StandardScaler()
X_train_n = sc.fit_transform(X_train[['Engine Size (L)']])
X_test_n = sc.transform(X_test[['Engine Size (L)']])

plt.subplot(1, 2, 2)
plt.hist(X_train_n, bins=20, color='magenta', edgecolor='black')
plt.xlabel('Engine Size (L) (skalirano)')
plt.ylabel('Frekvencija')
plt.title('Histogram nakon skaliranja')
plt.show()

# d) Izgradite linearni regresijski modeli. Ispišite u terminal dobivene parametre modela i
# povežite ih s izrazom 4.6.

import sklearn.linear_model as lm
 
linearModel = lm.LinearRegression()
linearModel.fit(X_train_n, y_train)

print("Koeficijenti modela:", linearModel.coef_)
print("Presjek (intercept) modela:", linearModel.intercept_)
