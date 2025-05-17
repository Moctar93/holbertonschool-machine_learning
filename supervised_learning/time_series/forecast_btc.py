import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.losses import MeanSquaredError

def load_dataset():
    """Charge les séquences et cibles depuis le fichier pré-traité."""
    data = np.load("btc_dataset.npz")
    X, y = data['X'], data['y']

    # Pour LSTM : reshape en (samples, timesteps, features)
    X = X[..., np.newaxis]  # Ajoute une dimension "features"
    return X, y

def create_tf_dataset(X, y, batch_size=64):
    """Crée un objet tf.data.Dataset (efficace pour l'entraînement TensorFlow)."""
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    ds = ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(input_shape):
    """Crée un modèle LSTM simple."""
    model = Sequential([
        LSTM(64, input_shape=input_shape),  # couche LSTM
        Dense(1)  # une seule sortie : le prix prédit
    ])
    model.compile(optimizer='adam', loss=MeanSquaredError())
    return model

if __name__ == "__main__":
    X, y = load_dataset()

    # Split train/validation
    split = int(0.8 * len(X))
    train_ds = create_tf_dataset(X[:split], y[:split])
    val_ds = create_tf_dataset(X[split:], y[split:])

    # Créer et entraîner le modèle
    model = build_model(input_shape=(X.shape[1], X.shape[2]))
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    # Sauvegarde
    model.save("btc_forecast_model.h5")

