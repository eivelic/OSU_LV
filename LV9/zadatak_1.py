import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from matplotlib import pyplot as plt

# ucitaj CIFAR-10 podatkovni skup
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# prikazi 9 slika iz skupa za ucenje
plt.figure()
for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.xticks([]),plt.yticks([])
    plt.imshow(X_train[i])

plt.show()

# pripremi podatke (skaliraj ih na raspon [0,1]])
X_train_n = X_train.astype('float32')/ 255.0
X_test_n = X_test.astype('float32')/ 255.0

# 1-od-K kodiranje
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# CNN mreza
model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(500, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

# definiraj listu s funkcijama povratnog poziva
my_callbacks = [
    keras.callbacks.TensorBoard(log_dir='logs/cnn_droput', update_freq=100)
]

model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

model.fit(X_train_n,
            y_train,
            epochs = 40,
            batch_size = 64,
            callbacks = my_callbacks,
            validation_split = 0.1)

score = model.evaluate(X_test_n, y_test, verbose=0)
print(f'Tocnost na testnom skupu podataka: {100.0*score[1]:.2f}')

# Skripta Zadatak_1.py učitava CIFAR-10 skup podataka. Ovaj skup sadrži
# 50000 slika u skupu za učenje i 10000 slika za testiranje. Slike su RGB i rezolucije su 32x32.
# Svakoj slici je pridružena jedna od 10 klasa ovisno koji je objekt prikazan na slici. Potrebno je:

# 1. Proučite dostupni kod. Od kojih se slojeva sastoji CNN mreža? Koliko ima parametara mreža?

# Ova CNN mreža se sastoji od tri konvolucijska sloja, svakog praćenog MaxPooling slojem, zatim sloja za izravnavanje (Flatten) i dva potpuno povezana sloja.
# Prvi konvolucijski sloj koristi 32 filtera, drugi 64, a treći 128, dok potpuno povezani sloj ima 500 neurona, a izlazni sloj 10 neurona (za 10 klasa CIFAR-10). 
# Ukupni broj parametara ove mreže iznosi 102495630, što uključuje sve filtere, neurone i njihove parametre u slojevima.

# 2. Pokrenite učenje mreže. Pratite proces učenja pomoću alata Tensorboard na sljedeći način.
# Pokrenite Tensorboard u terminalu pomoću naredbe:
# tensorboard –logdir=logs
# i zatim otvorite adresu http://localhost:6006/ pomoću web preglednika.

# Za procesa treniranja, preciznost na validacijskom skupu počinje opadati, dok funkcija gubitka s vremenom raste.

# Modificirajte skriptu iz prethodnog zadatka na način da na odgovarajuća mjesta u mrežu dodate droput slojeve.
# Prije pokretanja učenja promijenite Tensorboard funkciju povratnog poziva na način da informacije zapisuje u novi direktorij (npr. =/log/cnn_droput).
# Pratite tijek učenja. Kako komentirate utjecaj dropout slojeva na performanse mreže?

# S Dropout slojevima: Model je prisiljen učiti robusnije značajke jer mora koristiti različite skupove neurona na svakom koraku treninga.
# Točnost na treningu može biti nešto niža s dropout slojevima, jer model neće biti u mogućnosti iskoristiti cijeli skup neurona.

# Dodajte funkciju povratnog poziva za rano zaustavljanje koja će zaustaviti proces učenja nakon što se 5 uzastopnih epoha
# ne smanji prosječna vrijednost funkcije gubitka na validacijskom skupu.

from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True,
                               verbose=1)

# Što se događa s procesom učenja:
# 1. ako se koristi jako velika ili jako mala veličina serije?

# - Prevelika veličina serije može učiniti trening stabilnijim, ali sporijim i sklonijim pretreniranje,
#   dok mala veličina serije može ubrzati trening, ali ga čini nestabilnijim i manje preciznim.

# 2. ako koristite jako malu ili jako veliku vrijednost stope učenja?

# - Mala stopa učenja čini trening sporim, ali stabilnim, dok velika stopa može ubrzati trening,
#   ali uz rizik od nestabilnosti i loših rezultata.

# 3. ako izbacite određene slojeve iz mreže kako biste dobili manju mrežu?

# - Manja mreža može trenirati brže i smanjiti pretreniranje, ali uz rizik smanjenja
#   sposobnosti za učenje složenih obrazaca.

# 4. ako za 50% smanjite veličinu skupa za učenje?

# - Manji skup podataka ubrzava trening, ali može smanjiti generalizaciju i učiniti model sklonom pretreniranje.
