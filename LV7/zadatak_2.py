import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as Image
from sklearn.cluster import KMeans

# ucitaj sliku
img = Image.imread("test_2.jpg")

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

K=5
kmeans=KMeans(n_clusters=K, init='k-means++', random_state=0)
kmeans.fit(img_array)

centroids=kmeans.cluster_centers_
labels=kmeans.labels_

print('Centroidi boja:')
print(centroids)

# 3. Vrijednost svakog elementa slike originalne slike zamijeni s njemu pripadajućim centrom.

img_array_aprox=centroids[labels]
img_approx=np.reshape(img_array_aprox,(w,h,d))

plt.figure()
plt.title(f"Kvantizirana slika (K = {K})")
plt.imshow(img_approx)
plt.tight_layout()
plt.show()

# 4. Usporedite dobivenu sliku s originalnom. Mijenjate broj grupa K. 
# Komentirajte dobivene rezultate.

for K in [2, 4, 8, 16, 32]:
    kmeans = KMeans(n_clusters=K, init='random',random_state=0)
    kmeans.fit(img_array)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    img_aprox = np.reshape(centroids[labels], (w, h, d))

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img)
    axs[0].set_title("Original")
    axs[0].axis("off")
    axs[1].imshow(img_aprox)
    axs[1].set_title(f"K = {K}")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

# Komentar:
# S povećanjem broja klastera (K), kvaliteta rekonstrukcije slike se poboljšava jer se broj boja povećava, što omogućuje precizniju reprodukciju originalne slike.
# Ipak, rekonstrukcija s K-Means uvijek podrazumijeva gubitak određenih detalja jer ne možemo točno replicirati sve boje prisutne u originalu.

# 5. Primijenite postupak i na ostale dostupne slike.

# Komentar: Samo mijenjamo sliku u putanji u prethodnom zadatku.

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

img = Image.imread("test_2.jpg")

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

img = Image.imread("test_2.jpg")

n_clusters = 5
show_binary_images_for_clusters(img, n_clusters)

# Komentar:
# Binarne slike nam pomažu vidjeti kako su pikseli grupirani prema njihovim bojama. 
# Na primjer, u nekim slikama, klasteri mogu odgovarati različitim objektima ili područjima slike (npr. pozadina, nebo, predmeti).
