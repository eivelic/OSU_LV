import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

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
                    label=cl)


# ucitaj podatke
data = pd.read_csv("Social_Network_Ads.csv")
print(data.info())

data.hist()
plt.show()

# dataframe u numpy
X = data[["Age","EstimatedSalary"]].to_numpy()
y = data["Purchased"].to_numpy()

# podijeli podatke u omjeru 80-20%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify=y, random_state = 10)

# skaliraj ulazne velicine
sc = StandardScaler()
X_train_n = sc.fit_transform(X_train)
X_test_n = sc.transform((X_test))

# Model logisticke regresije
LogReg_model = LogisticRegression(penalty=None) 
LogReg_model.fit(X_train_n, y_train)

# Evaluacija modela logisticke regresije
y_train_p = LogReg_model.predict(X_train_n)
y_test_p = LogReg_model.predict(X_test_n)

print("Logisticka regresija: ")
print("Tocnost train: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
print("Tocnost test: " + "{:0.3f}".format((accuracy_score(y_test, y_test_p))))

# granica odluke pomocu logisticke regresije
plot_decision_regions(X_train_n, y_train, classifier=LogReg_model)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title("Tocnost: " + "{:0.3f}".format((accuracy_score(y_train, y_train_p))))
plt.tight_layout()
plt.show()

# Skripta zadatak_1.py učitava Social_Network_Ads.csv skup podataka [2].
# Ovaj skup sadrži podatke o korisnicima koji jesu ili nisu napravili kupovinu za prikazani oglas.
# Podaci o korisnicima su spol, dob i procijenjena plaća. Razmatra se binarni klasifikacijski
# problem gdje su dob i procijenjena plaća ulazne veličine, dok je kupovina (0 ili 1) izlazna
# veličina. Za vizualizaciju podatkovnih primjera i granice odluke u skripti je dostupna funkcija
# plot_decision_region [1]. Podaci su podijeljeni na skup za učenje i skup za testiranje modela
# u omjeru 80%-20% te su standardizirani. Izgrađen je model logističke regresije te je izračunata
# njegova točnost na skupu podataka za učenje i skupu podataka za testiranje.

# Izradite algoritam KNN na skupu podataka za učenje (uz K = 5). Izračunajte točnost klasifikacije na skupu podataka za učenje i skupu podataka za testiranje. Usporedite dobivene rezultate s rezultatima logističke regresije. Što primjećujete vezano uz dobivenu
# granicu odluke KNN modela?

KNN = KNeighborsClassifier(n_neighbors=5)
KNN.fit(X_train_n, y_train)    

y_train_knn_p = KNN.predict(X_train_n)  
y_test_knn_p = KNN.predict(X_test_n)   

print("Test:", "{:0.3f}".format((accuracy_score(y_test, y_test_knn_p))))
print("Train:", "{:0.3f}".format((accuracy_score(y_train, y_train_knn_p))))

plot_decision_regions(X_train_n, y_train, classifier=KNN)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title("K=5")
plt.legend(loc='upper left')
plt.show()

# Kako izgleda granica odluke kada je K = 1 i kada je K = 100?

# K = 1:
# Model uzima samo najbližeg susjeda da bi donio odluku o klasi novog podatka.
# Može savršeno klasificirati podatke iz skupa za učenje (trenutni podaci), što znači da će točnost na trening skupu biti vrlo visoka (često 100%).
# Model previše prati šum u podacima (overfitting). Na test skupu može imati lošiju točnost jer se ne generalizira dobro.

# K = 100:
# Model uzima 100 najbližih susjeda.
# Model je glatkiji i stabilniji, bolje generalizira te je manja je šansa za overfitting.
# Ako je broj susjeda velik u odnosu na broj podataka, granica može biti previše zaglađena što može dovesti do underfittinga.

# Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametra K algoritma KNN za podatke iz Zadatka 1.

param_grid = {'n_neighbors': np.arange(1, 101)}  # testiramo K od 1 do 100
knn = KNeighborsClassifier()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_n, y_train)

print("Optimalni K:", grid_search.best_params_['n_neighbors'])
print("Najbolja točnost na treningu (CV):", "{:0.3f}".format(grid_search.best_score_))

# Na podatke iz Zadatka 1 primijenite SVM model koji koristi RBF kernel funkciju
# te prikažite dobivenu granicu odluke. Mijenjajte vrijednost hiperparametra C i γ.
# Kako promjena ovih hiperparametara utječe na granicu odluke te pogrešku na skupu podataka za testiranje?
# Mijenjajte tip kernela koji se koristi. Što primjećujete?

svm_rbf = svm.SVC(kernel='rbf', random_state=10)
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001]
}
grid_search = GridSearchCV(estimator=svm_rbf, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_n, y_train)
print(grid_search.best_params_)

svm_best = grid_search.best_estimator_

y_train_svm_p = svm_best.predict(X_train_n)
y_test_svm_p = svm_best.predict(X_test_n)

print("Test:", "{:0.3f}".format((accuracy_score(y_test, y_test_svm_p))))
print("Train:", "{:0.3f}".format((accuracy_score(y_train, y_train_svm_p))))

plot_decision_regions(X_train_n, y_train, classifier=svm_best)
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.legend(loc='upper left')
plt.title('SVM')
plt.tight_layout()
plt.show()

# Manji C i gamma daju široke granice (manji rizik od overfittinga, ali može underfitati).
# Veći C i gamma daju kompleksne, "savijene" granice koje prate trening podatke (rizično za overfitting).
# Promjenom kernela (npr. linear, poly) dobivamo potpuno različite granice.

# Pomoću unakrsne validacije odredite optimalnu vrijednost hiperparametra C i γ
# algoritma SVM za problem iz Zadatka 1.

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

svm_model = svm.SVC(random_state=10)
grid_search = GridSearchCV(estimator=svm_model,
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy')

grid_search.fit(X_train_n, y_train)

print("Najbolji hiperparametri:", grid_search.best_params_)
