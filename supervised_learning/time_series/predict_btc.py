import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Charger les données et le modèle
data = np.load("btc_dataset.npz")
X, y_true = data["X"], data["y"]

# Reshape pour correspondre à l'entrée du modèle
X = X.reshape((X.shape[0], X.shape[1], 1))

# Charger le modèle
model = keras.models.load_model("btc_forecast_model.h5")

# Faire des prédictions
y_pred = model.predict(X)

# Afficher les 100 dernières vraies valeurs vs prédictions
n = 100
plt.figure(figsize=(12, 6))
plt.plot(y_true[-n:], label="Valeurs réelles")
plt.plot(y_pred[-n:], label="Prédictions")
plt.xlabel("Minute")
plt.ylabel("Prix (normalisé)")
plt.title("Prédiction du prix du BTC pour les 100 dernières minutes")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

