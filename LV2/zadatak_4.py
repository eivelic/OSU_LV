#Napišite program koji će kreirati sliku koja sadrži četiri kvadrata crne odnosno
#bijele boje (vidi primjer slike 2.4 ispod). Za kreiranje ove funkcije koristite numpy funkcije
#zeros i ones kako biste kreirali crna i bijela polja dimenzija 50x50 piksela. Kako biste ih složili
#u odgovarajući oblik koristite numpy funkcije hstack i vstack.

import numpy as np
import matplotlib.pyplot as plt

black_square = np.zeros((50,50))
white_square = np.ones((50,50))

top_row = np.hstack((black_square, white_square))
bottom_row = np.hstack((white_square, black_square))

img = np.vstack((top_row, bottom_row))

plt.imshow(img, cmap="gray")
plt.show()
