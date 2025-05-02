# Napišite skriptu koja ce učitati izgradenu mrežu iz zadatka 1. 
# Nadalje, skripta treba učitati sliku test.png sa diska. 
# Dodajte u skriptu kod koji ce prilagoditi sliku za mrežu, klasiﬁcirati sliku pomoću 
# izgređene mreže te ispisati rezultat u terminal. 
# Promijenite sliku pomoću nekog graﬁčkog alata (npr. pomocu Windows Paint-a nacrtajte broj 2)
# i ponovo pokrenite skriptu.
# Komentirajte dobivene rezultate za različite napisane znamenke.

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

model = load_model("model.keras")

img = load_img("C:/Users/user/Desktop/test.png", color_mode="grayscale", target_size=(28, 28))
img_array = img_to_array(img)

plt.imshow(img_array.squeeze(), cmap="gray")
plt.title("Učitana slika")
plt.axis("off")
plt.show()

img_array = 255 - img_array

img_array = img_array.astype("float32") / 255.0

img_array = img_array.reshape(1, 784)

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)

print(f"Predikcija modela: {predicted_class}")
