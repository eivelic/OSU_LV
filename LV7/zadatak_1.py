import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering


def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe
    elif flagc == 2:
        random_state = 148
        X,y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe 
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers = 4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe  
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

# generiranje podatkovnih primjera
X = generate_data(500, 5)

# prikazi primjere u obliku dijagrama rasprsenja
plt.figure()
plt.scatter(X[:,0],X[:,1])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('podatkovni primjeri')
plt.show()

# Skripta zadatak_1.py sadrži funkciju generate_data koja služi za generiranje
# umjetnih podatkovnih primjera kako bi se demonstriralo grupiranje. Funkcija prima cijeli broj
# koji definira željeni broju uzoraka u skupu i cijeli broj od 1 do 5 koji definira na koji način će
# se generirati podaci, a vraća generirani skup podataka u obliku numpy polja pri čemu su prvi i
# drugi stupac vrijednosti prve odnosno druge ulazne veličine za svaki podatak. Skripta generira
# 500 podatkovnih primjera i prikazuje ih u obliku dijagrama raspršenja.

# 1. Pokrenite skriptu. Prepoznajete li koliko ima grupa u generiranim podacima? Mijenjajte
# način generiranja podataka.
# Komentar:
# flagc == 1 će generirati 3 grupe.
# flagc == 2 također generira 3 grupe, ali s transformacijama.
# flagc == 3 generira 4 grupe s različitim gustoćama.
# flagc == 4 stvara 2 grupe u obliku krugova.
# flagc == 5 stvara 2 grupe u obliku polumjeseca.

# 2. Primijenite metodu K srednjih vrijednosti te ponovo prikažite primjere, ali svaki primjer
# obojite ovisno o njegovoj pripadnosti pojedinoj grupi. Nekoliko puta pokrenite programski
# kod. Mijenjate broj K. Što primjećujete?

def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    y_kmeans = kmeans.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=30)

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'K-Means clustering s {n_clusters} klastera')
    plt.show()

for n_clusters in range(2, 6):
    kmeans_clustering(X, n_clusters)

# Komentar:
# Ako odaberemo previše klastera (npr. 4/5), K-Means će pretjerano podijeliti podatke, što može dovesti do podjela koje nemaju smisla. 
# Na grafu se vidi previše podjela, čak iako podaci imaju samo nekoliko stvarnih grupa.
# Ako odaberemo premalo klastera (npr. 2), K-Means neće moći ispravno grupirati podatke, što može rezultirati time da se podaci iz različitih stvarnih grupa spoje u jedan klaster.
# Kao nekakvu dobru podjelu, tj. za optimalan broj klastera (npr. 3), K-Means može točno prepoznati stvarne grupe, a podaci će biti pravilno grupirani.

# 3. Mijenjajte način definiranja umjetnih primjera te promatrajte rezultate grupiranja podataka
# (koristite optimalni broj grupa). Kako komentirate dobivene rezultate?

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans

# Funkcija za generiranje podataka
def generate_data(n_samples, flagc):
    # 3 grupe
    if flagc == 1:
        random_state = 365
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    
    # 3 grupe s transformacijom
    elif flagc == 2:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples, random_state=random_state)
        transformation = [[0.60834549, -0.63667341], [-0.40887718, 0.85253229]]
        X = np.dot(X, transformation)

    # 4 grupe s različitim gustoćama
    elif flagc == 3:
        random_state = 148
        X, y = make_blobs(n_samples=n_samples,
                        centers=4,
                        cluster_std=np.array([1.0, 2.5, 0.5, 3.0]),
                        random_state=random_state)
    # 2 grupe u obliku krugova
    elif flagc == 4:
        X, y = make_circles(n_samples=n_samples, factor=.5, noise=.05)
    
    # 2 grupe u obliku polumjeseca
    elif flagc == 5:
        X, y = make_moons(n_samples=n_samples, noise=.05)
    
    else:
        X = []
        
    return X

def kmeans_clustering(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    y_kmeans = kmeans.fit_predict(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', s=30)

    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X')
    
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(f'K-Means clustering s {n_clusters} klastera')
    plt.show()

for flagc in range(1, 6):
    X = generate_data(500, flagc)
    
    print(f"\nGenerirani podaci za flagc = {flagc}")
    for n_clusters in range(2, 6):
        kmeans_clustering(X, n_clusters)

# Komentar:
# Ovisno o načinu generiranja podataka, K-Means može ili točno prepoznati stvarne grupe ili pretjerano podijeliti podatke.
# Optimalan broj klastera mora odgovarati stvarnim grupama u podacima kako bi klasteriranje bilo precizno.
