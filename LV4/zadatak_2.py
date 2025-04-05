# Na temelju rješenja prethodnog zadatka izradite model koji koristi i kategoričku
# varijablu "Fuel Type" kao ulaznu veličinu. Pri tome koristite 1-od-K kodiranje kategoričkih
# veličina. Radi jednostavnosti nemojte skalirati ulazne veličine. Komentirajte dobivene rezultate.
# Kolika je maksimalna pogreška u procjeni emisije C02 plinova u g/km? 
# O kojem se modelu vozila radi?

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import max_error
import numpy as np

df = pd.read_csv('data_C02_emission.csv')

ohe = OneHotEncoder()
X_encoded = ohe.fit_transform(df[['Fuel Type']]).toarray()

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns.drop('CO2 Emissions (g/km)')
X_numeric = df[numeric_columns]
X_combined = np.concatenate([X_numeric.values, X_encoded], axis=1)
y = df['CO2 Emissions (g/km)'] 

X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=1)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
max_err = max_error(y_test, y_pred)
max_idx = np.argmax(np.abs(y_test - y_pred))

row_with_max_error = df.iloc[y_test.index[max_idx]]

print(f"Maksimalna pogreška u procjeni CO2 emisije: {max_err:.2f} g/km")
print("Podaci o vozilu s najvećom pogreškom:")
print(row_with_max_error)
