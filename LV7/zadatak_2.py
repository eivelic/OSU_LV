import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("imgs\\imgs\\test_1.jpg")

# prikazi originalnu sliku
plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

# pretvori vrijednosti elemenata slike u raspon 0 do 1
img = img.astype(np.float64) / 255

# transfromiraj sliku u 2D numpy polje (jedan red su RGB komponente elementa slike)
w,h,d = img.shape
img_array = np.reshape(img, (w*h, d))

# rezultatna slika
img_array_aprox = img_array.copy()

# 1. Otvorite skriptu zadatak_2.py. Ova skripta učitava originalnu RGB sliku test_1.jpg
# te ju transformira u podatkovni skup pri čemu je n broj elemenata slike, a m je jednak 3. 
# Koliko je različitih boja prisutno u ovoj slici?

img_array_rounded = np.round(img_array, decimals=3)
unique_colors = np.unique(img_array_rounded, axis=0)
print(f"Broj različitih boja u slici: {unique_colors.shape[0]}")

# 2. Primijenite algoritam K srednjih vrijednosti koji ce pronaći grupe u RGB vrijednostima
# elemenata originalne slike.
n_clusters = 5
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(img_array)

labels = kmeans.predict(img_array)

cluster_centers = kmeans.cluster_centers_

img_array_aprox = cluster_centers[labels]
img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

plt.figure()
plt.title("Originalna slika")
plt.imshow(img)
plt.tight_layout()
plt.show()

plt.figure()
plt.title(f"Slika s {n_clusters} klastera")
plt.imshow(img_array_aprox)
plt.tight_layout()
plt.show()

# 3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajućim centrom.

plt.figure()
plt.title(f"Slika s K-Means centrom za svaki piksel ({n_clusters} klastera)")
plt.imshow(img_array_aprox)
plt.tight_layout()
plt.show()

# 4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. 
# Komentirajte dobivene rezultate.
# Komentar:
# S povećanjem broja klastera (K), kvaliteta rekonstrukcije slike se poboljšava jer se broj boja povećava, što omogućuje precizniju reprodukciju originalne slike.
# Ipak, rekonstrukcija s K-Means uvijek podrazumijeva gubitak određenih detalja jer ne možemo točno replicirati sve boje prisutne u originalu.

# 5. Primijenite postupak i na ostale dostupne slike.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans
import os

def apply_kmeans_and_plot(img, n_clusters):
    img = img.astype(np.float64) / 255

    img = np.clip(img, 0, 1)

    w, h, d = img.shape
    img_array = np.reshape(img, (w*h, d))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img_array)
    
    labels = kmeans.predict(img_array)
    
    cluster_centers = kmeans.cluster_centers_

    img_array_aprox = cluster_centers[labels]
    img_array_aprox = np.reshape(img_array_aprox, (w, h, d))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Originalna slika")
    plt.imshow(img)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f"Slika s {n_clusters} klastera")
    plt.imshow(img_array_aprox)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def process_images_in_directory(directory, n_clusters=5):
    for filename in os.listdir(directory):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.imread(os.path.join(directory, filename))

            print(f"Obrađujem sliku: {filename}")
            apply_kmeans_and_plot(img, n_clusters)

image_directory = "imgs\\imgs\\"

process_images_in_directory(image_directory, n_clusters=5)

# 6. Grafički prikažite ovisnost J o broju grupa K. Koristite atribut inertia objekta klase KMeans.
# Možete li uočiti lakat koji upućuje na optimalni broj grupa?

def plot_inertia_vs_k(img, max_k=10):
    img = img.astype(np.float64) / 255

    img = np.clip(img, 0, 1)

    w, h, d = img.shape
    img_array = np.reshape(img, (w*h, d))

    inertias = []

    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(img_array)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 6))
    plt.plot(range(1, max_k+1), inertias, marker='o', linestyle='-', color='b')
    plt.title("Ovisnost inertije o broju klastera K")
    plt.xlabel("Broj klastera K")
    plt.ylabel("Inertija")
    plt.tight_layout()
    plt.show()

img = Image.imread("imgs\\imgs\\test_1.jpg")

plot_inertia_vs_k(img, max_k=10)

# 7. Elemente slike koji pripadaju jednoj grupi prikažite kao zasebnu binarnu sliku.
# Što primjećujete?

def show_binary_images_for_clusters(img, n_clusters):
    img = img.astype(np.float64) / 255
    img = np.clip(img, 0, 1)

    w, h, d = img.shape
    img_array = np.reshape(img, (w*h, d))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(img_array)

    labels = kmeans.predict(img_array)

    plt.figure(figsize=(15, 8))

    for k in range(n_clusters):
        binary_img = (labels == k).reshape(w, h)

        plt.subplot(1, n_clusters, k+1)
        plt.imshow(binary_img, cmap='gray')
        plt.title(f"Klaster {k+1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

img = Image.imread("imgs\\imgs\\test_1.jpg")

n_clusters = 5
show_binary_images_for_clusters(img, n_clusters)

# Komentar:
# Binarne slike nam pomažu vidjeti kako su pikseli grupirani prema njihovim bojama. 
# Na primjer, u nekim slikama, klasteri mogu odgovarati različitim objektima ili područjima slike (npr. pozadina, nebo, predmeti).
