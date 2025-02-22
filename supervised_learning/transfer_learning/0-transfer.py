#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

# Vérifier la version de TensorFlow
print("Version de TensorFlow :", tf.__version__)

# Vérifier la disponibilité du GPU
print("GPU disponible :", tf.config.list_physical_devices('GPU'))

# Fonction pour prétraiter les données
def preprocess_data(X, Y):
    """
    Prétraite les données CIFAR-10 pour le modèle.

    Arguments :
        X -- numpy.ndarray de forme (m, 32, 32, 3) contenant les images CIFAR-10.
        Y -- numpy.ndarray de forme (m,) contenant les étiquettes CIFAR-10.

    Retourne :
        X_p -- numpy.ndarray contenant les images prétraitées.
        Y_p -- numpy.ndarray contenant les étiquettes prétraitées (one-hot encoding).
    """
    # Redimensionner les images à 224x224 (taille attendue par MobileNet)
    X = tf.image.resize(X, [224, 224])
    
    # Normaliser les données (selon les attentes de MobileNet)
    X = tf.keras.applications.mobilenet.preprocess_input(X)
    
    # Encoder les labels en one-hot encoding
    Y = to_categorical(Y, 10)
    
    return X, Y

# Fonction pour construire le modèle
def build_model():
    """
    Construit un modèle de classification basé sur MobileNet.

    Retourne :
        model -- Un modèle Keras compilé et prêt à être entraîné.
    """
    # Charger MobileNet sans les couches fully connected
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Geler les couches du modèle de base
    for layer in base_model.layers:
        layer.trainable = False
    
    # Ajouter des couches personnalisées
    x = base_model.output
    x = GlobalAveragePooling2D()(x)  # Réduire la dimension spatiale
    x = Dense(1024, activation='relu')(x)  # Couche fully connected
    predictions = Dense(10, activation='softmax')(x)  # Couche de sortie pour 10 classes
    
    # Créer le modèle final
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

# Point d'entrée du script
if __name__ == "__main__":
    # 1. Charger les données CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    # 2. Prétraiter les données
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_test = preprocess_data(X_test, Y_test)
    
    # 3. Construire et compiler le modèle
    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 4. Entraîner le modèle
    print("Début de l'entraînement...")
    history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)
    
    # 5. Sauvegarder le modèle
    model.save("cifar10.h5")
    print("Modèle sauvegardé sous 'cifar10.h5'.")
