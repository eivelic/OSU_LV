# Skripta zadatak_3.py ucitava sliku 'road.jpg'.
# Manipulacijom odgovarajuce numpy matrice pokušajte:

import numpy as np
import matplotlib.pyplot as plt

img = plt.imread("road.jpg")

#a) posvijetliti sliku
bright_img = np.clip(img*1.9, 0, 255).astype(np.uint8)
plt.imshow(bright_img)
plt.show()

#b) prikazati samo drugu četvrtinu slike po širini
h, w = img.shape[:2]
cropped_img = img[:, w//4:w//2]
plt.imshow(cropped_img)
plt.show()

#c) zarotirati sliku za 90 stupnjeva u smjeru kazaljke na satu
rotated_img = np.rot90(img, k=-1)
plt.imshow(rotated_img)
plt.show()

#d) zrcaliti sliku
mirrored_img = img[:, ::-1]
plt.imshow(mirrored_img)
plt.show()
