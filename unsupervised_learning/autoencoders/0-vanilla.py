#!/usr/bin/env python3
"""
Implement an autoencoder model.
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Create an autoencoder.
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = inputs
    for units in hidden_layers:
        encoded = keras.layers.Dense(units, activation='relu')(encoded)
    latent = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    decoded_input = keras.Input(shape=(latent_dims,))
    decoded = decoded_input
    for units in reversed(hidden_layers):
        decoded = keras.layers.Dense(units, activation='relu')(decoded)
    outputs = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    encoder = keras.Model(inputs, latent, name='encoder')
    decoder = keras.Model(decoded_input, outputs, name='decoder')
    auto = keras.Model(inputs, decoder(encoder(inputs)), name='autoencoder')

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
