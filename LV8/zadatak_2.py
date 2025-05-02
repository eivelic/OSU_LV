# Napišite skriptu koja će učitati izgrađenu mrežu iz zadatka 1 i MNIST skup podataka. 
# Pomoću matplotlib biblioteke potrebno je prikazati nekoliko loše klasiﬁ ciranih slika iz
# skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predviđenu mrežom.

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

model = load_model("model.keras")

(_, _), (x_test, y_test) = mnist.load_data()

x_test_scaled = x_test.astype("float32") / 255
x_test_flat = x_test_scaled.reshape((-1, 784))

predictions = model.predict(x_test_flat)
predicted_classes = np.argmax(predictions, axis=1)

incorrect_indices = np.where(predicted_classes != y_test)[0]
print(f"Broj pogrešnih klasifikacija: {len(incorrect_indices)}")

plt.figure(figsize=(12, 4))
for i in range(5):
    idx = incorrect_indices[i]
    plt.subplot(1, 5, i + 1)
    plt.imshow(x_test[idx], cmap="gray")
    plt.title(f"Stvarno: {y_test[idx]}\nPredikcija: {predicted_classes[idx]}")
    plt.axis("off")

plt.tight_layout()
plt.show()
