import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split

labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

def plot_decision_regions(X, y, classifier, resolution=0.02):
    plt.figure()
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    edgecolor = 'w',
                    label=labels[cl])

# ucitaj podatke
df = pd.read_csv("penguins.csv")

# izostale vrijednosti po stupcima
print(df.isnull().sum())

# spol ima 11 izostalih vrijednosti; izbacit cemo ovaj stupac
df = df.drop(columns=['sex'])

# obrisi redove s izostalim vrijednostima
df.dropna(axis=0, inplace=True)

# kategoricka varijabla vrsta - kodiranje
df['species'].replace({'Adelie' : 0,
                        'Chinstrap' : 1,
                        'Gentoo': 2}, inplace = True)

print(df.info())

# izlazna velicina: species
output_variable = ['species']

# ulazne velicine: bill length, flipper_length
input_variables = ['bill_length_mm',
                    'flipper_length_mm']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()

# podjela train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

# Skripta zadatak_2.py učitava podatkovni skup Palmer Penguins [1]. 
# Ovaj podatkovni skup sadrži mjerenja provedena na tri različite vrste pingvina ('Adelie', 
# 'Chins-trap', 'Gentoo') na tri različita otoka u području Palmer Station, Antarktika. Vrsta pingvina
# odabrana je kao izlazna veličina i pri tome su klase označene s cjelobrojnim vrijednostima
# 0, 1 i 2. Ulazne veličine su duljina kljuna ('bill_length_mm') i duljina peraje u mm ('flip-
# per_length_mm'). Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna
# funkcija plot_decision_region.

# a) Pomoću stupčastog dijagrama prikažite koliko primjera postoji za svaku klasu
# (vrstu pingvina) u skupu podataka za učenje i skupu podataka za testiranje. 
# Koristite numpy funkciju unique.

train_classes, train_counts = np.unique(y_train, return_counts=True)
test_classes, test_counts = np.unique(y_test, return_counts=True)

x = np.arange(len(train_classes))
width = 0.35

fig, ax = plt.subplots()
bars1 = ax.bar(x - width/2, train_counts, width, label='Skup za učenje', color='skyblue')
bars2 = ax.bar(x + width/2, test_counts, width, label='Skup za testiranje', color='lightgreen')

ax.set_xlabel('Vrsta pingvina')
ax.set_ylabel('Broj primjera')
ax.set_title('Broj primjera po klasama u skupu za učenje i testiranju')
ax.set_xticks(x)
ax.set_xticklabels(['Adelie (0)', 'Chinstrap (1)', 'Gentoo (2)'])
ax.legend()
plt.tight_layout()
plt.show()

# b) Izgradite model logističke regresije pomoću scikit-learn biblioteke na temelju skupa
# podataka za učenje.

from sklearn.linear_model import LogisticRegression

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

# c) Pronađite u atributima izgrađenog modela parametre modela.
# Koja je razlika u odnosu na binarni klasifikacijski problem iz prvog zadatka?

print("Koeficijenti: ", LogRegression_model.coef_)
print("Intercept: ", LogRegression_model.intercept_)

# coef_ - matrica oblika (n_classes, n_features), znači da imamo po jedan vektor koeficijenata za svaku klasu, 
# jer se u višeklasnoj logističkoj regresiji koristi "one-vs-rest" pristup.
# intercept_ - jedno presretanje za svaku klasu.

# d) Pozovite funkciju plot_decision_region pri čemu joj predajte podatke za učenje i
# izgrađeni model logističke regresije. Kako komentirate dobivene rezultate?

plot_decision_regions(X_train, y_train.ravel(), LogRegression_model)
plt.xlabel('Duljina kljuna (mm)')
plt.ylabel('Duljina peraje (mm)')
plt.title('Granice odluke logističke regresije (skup za učenje)')
plt.legend()
plt.tight_layout()
plt.show()

# Granice odluke podijelile su prostor u tri regije, svaka pripada jednoj vrsti pingvina (Adelie, Chinstrap, Gentoo).
# Većina točaka se nalazi unutar ispravnih regija, što znači da model dobro razdvaja klase na temelju duljine kljuna i peraje.
# Međutim, mogu se primijetiti neka preklapanja, posebno između vrsta Adelie i Chinstrap, što sugerira 
# da te dvije vrste imaju sličnija mjerenja i teže ih je razdvojiti samo pomoću ove dvije značajke.
# Vrsta Gentoo je najbolje razdvojena — ima jasnu granicu prema ostalima, što može značiti da ima dulju peraju ili specifičnu duljinu kljuna.

# e) Provedite klasifikaciju skupa podataka za testiranje pomoću izgrađenog modela logističke regresije.
# Izračunajte i prikažite matricu zabune na testnim podacima. Izračunajte točnost.
# Pomoću classification_report funkcije izračunajte vrijednost četiri glavne metrike na skupu podataka za testiranje.

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report

y_test_p = LogRegression_model.predict(X_test)
disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print("Točnost: ", accuracy_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p))

# f) Dodajte u model još ulaznih veličina. 
# Što se događa s rezultatima klasifikacije na skupu podataka za testiranje?

input_variables = ['bill_length_mm', 'flipper_length_mm', 'bill_depth_mm', 'body_mass_g']

X = df[input_variables].to_numpy()
y = df[output_variable].to_numpy().ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 123)

LogRegression_model = LogisticRegression()
LogRegression_model.fit(X_train, y_train)

y_test_p = LogRegression_model.predict(X_test)

disp = ConfusionMatrixDisplay(confusion_matrix(y_test , y_test_p))
disp.plot()
plt.show()

print("Točnost: ", accuracy_score(y_test, y_test_p))
print(classification_report(y_test, y_test_p))

# Model ima više informacija za razliku klasa pa je očekivano poboljšanje točnosti.
# Posebno korisno ako dodatne značajke bolje razlikuju slične klase (npr. Adelie i Chinstrap).
