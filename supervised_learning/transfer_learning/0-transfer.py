#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

# Fonction pour prétraiter les données
def preprocess_data(X, Y):
    X_p = tf.image.resize(X, [224, 224])
    X_p = tf.keras.applications.preprocess_input(X)
    X_p = to_categocal(Y, 10)
    return X_p, Y_p

#fonction pour construire le model
def build_model():
    base_model = MobileNet(weight='imageNet', include_top=False, input_shape=(224, 224, 3))
    for layer in base_mode.layers:
        layer.tranable = False

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        predictions = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        return model

#point d'entrée du script du model
if __name__ == "__main__":
    # 1.charger les données CIFAR-10
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
    
    # 2.prétraiter les données
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_test, Y_train = preprocess_data(X_test, Y_test)
    
    # 3.construire et compiler le model
    model = build_model()
    model.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 4.entrainer le model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=32)
    # 5.sauvegarder le model
    model.save(cifar10.h5)
