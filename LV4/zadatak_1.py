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

# e) Izvršite procjenu izlazne veličine na temelju ulaznih veličina skupa za testiranje. Prikažite
# pomoću dijagrama raspršenja odnos između stvarnih vrijednosti izlazne veličine i procjene
# dobivene modelom.

y_test_p = linearModel.predict(X_test_n)
plt.scatter(y_test, y_test_p, color='violet')
plt.xlabel("Stvarna vrijednost")
plt.ylabel("Predviđena vrijednost")
plt.title("Stvarne/predviđene vrijednosti CO2")
plt.show()

# f) Izvršite vrednovanje modela na način da izračunate vrijednosti regresijskih metrika na
# skupu podataka za testiranje.

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import numpy as np

mse = mean_squared_error(y_test, y_test_p)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_test_p)
mape = mean_absolute_percentage_error(y_test, y_test_p)
r2_determination = r2_score(y_test, y_test_p)

print(f"Srednja kvadratna pogreška (engl. mean squared error - MSE): {mse}")
print(f"Korijen iz srednje kvadratne pogreške (engl. root mean squared error - RMSE): {rmse}")
print(f"Srednja apsolutna pogreška (engl. mean absolute error - MAE): {mae}")
print(f"Srednja apsolutna postotna pogreška (engl. mean absolute percentage error - MAPE): {mape}")
print(f"Koeficijent determinacije R2 pokazuje koliko je varijacija u podacima obuhvaćeno modelom: {r2_determination}")

# g) Što se događa s vrijednostima evaluacijskih metrika na testnom skupu kada mijenjate broj
# ulaznih veličina?

X_full = df[numeric_columns.drop('CO2 Emissions (g/km)')]

X_train_full, X_test_full, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=1)

sc_full = StandardScaler()
X_train_full_n = sc_full.fit_transform(X_train_full)
X_test_full_n = sc_full.transform(X_test_full)

linearModel_full = lm.LinearRegression()
linearModel_full.fit(X_train_full_n, y_train)

y_test_full_p = linearModel_full.predict(X_test_full_n)

mse_full = mean_squared_error(y_test, y_test_full_p)
rmse_full = np.sqrt(mse_full)
mae_full = mean_absolute_error(y_test, y_test_full_p)
mape_full = mean_absolute_percentage_error(y_test, y_test_full_p)
r2_full = r2_score(y_test, y_test_full_p)

print("\nEvaluacija s više ulaznih značajki\n")
print(f"MSE: {mse_full}")
print(f"RMSE: {rmse_full}")
print(f"MAE: {mae_full}")
print(f"MAPE: {mape_full}")
print(f"R²: {r2_full}")

# Kada povećamo broj ulaznih značajki, regresijski model dobiva više informacija i bolje može naučiti odnose između ulaza i ciljne vrijednosti. 
# To često dovodi do poboljšanja regresijskih metrika (niži MSE, RMSE, MAE, MAPE i viši R²).
