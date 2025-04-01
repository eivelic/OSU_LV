import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# Skripta zadatak_1.py generira umjetni binarni klasifikacijski problem s dvije
# ulazne velicine. Podaci su podijeljeni na skup za učenje i skup za testiranje modela.

# a) Prikažite podatke za učenje u x1 − x2 ravnini matplotlib biblioteke pri čemu podatke obojite
# s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
# marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
# cmap kojima je moguće definirati boju svake klase
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', label='Trening podaci', edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test podaci', edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Podaci za učenje i testiranje')
plt.legend()
plt.show()

# b) Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa podataka
# za učenje
from sklearn.linear_model import LogisticRegression

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c) Pronađite u atributima izgrađenog modela parametre modela. Prikažite granicu odluke
# naučenog modela u ravnini x1 − x2 zajedno s podacima za učenje. 
# Napomena: granica odluke u ravnini x1 − x2 definirana je kao krivulja: θ0 + θ1x1 + θ2x2 = 0.
theta_0, theta_1, theta_2 = LogRegression_model.intercept_[0], LogRegression_model.coef_[0, 0], LogRegression_model.coef_[0, 1]

print(f"Parametri modela:")
print(f"θ0 (intercept): {theta_0}")
print(f"θ1 (coef_x1): {theta_1}")
print(f"θ2 (coef_x2): {theta_2}")

x1_vals = np.linspace(X_train[:, 0].min(), X_train[:, 0].max(), 100)
x2_vals = -(theta_0 + theta_1 * x1_vals) / theta_2

plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='RdYlBu', label='Trening podaci', edgecolors='k')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='coolwarm', marker='x', label='Test podaci', edgecolors='k')
plt.plot(x1_vals, x2_vals, label='Granica odluke', color='black', linestyle='--')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Granica odluke i podaci za učenje')
plt.legend()
plt.show()

# d) Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke
# regresije. Izračunajte i prikažite matricu zabune na testnim podacima. Izračunate točnost,
# preciznost i odziv na skupu podataka za testiranje.
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, ConfusionMatrixDisplay

y_pred = LogRegression_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title("Matrica zabune")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Točnost: {accuracy:.4f}")
print(f"Preciznost: {precision:.4f}")
print(f"Odziv: {recall:.4f}")

# e) Prikažite skup za testiranje u ravnini x1 − x2. Zelenom bojom oznacite dobro klasificirane
# primjere dok pogrešno klasificirane primjere označite crnom bojom.
plt.figure(figsize=(10, 6))
correct = y_test == y_pred
plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', label='Dobro klasificirani', edgecolors='k')

incorrect = y_test != y_pred
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='black', label='Pogrešno klasificirani', edgecolors='k')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Testni podaci s označenim klasifikacijama')
plt.legend()
plt.show()
