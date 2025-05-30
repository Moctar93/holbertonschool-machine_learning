#!/usr/bin/env python3
"""
Convolutional Autoencoder
"""

import tensorflow.keras as keras


def build_encoder(input_dims, filters):
    """
    Building encoder.
    """
    encoder_input = keras.layers.Input(shape=input_dims)

    x = encoder_input
    for f_dims in filters:
        x = keras.layers.Conv2D(filters=f_dims, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.MaxPool2D(pool_size=(2, 2), padding='same')(x)

    return keras.Model(inputs=encoder_input, outputs=x)


def build_decoder(latent_dims, filters, input_dims):
    """
    Building decoder.
    """

    decoder_input = keras.layers.Input(shape=latent_dims)

    x = decoder_input
    for f_dims in reversed(filters[1:]):
        x = keras.layers.Conv2D(filters=f_dims, kernel_size=(3, 3),
                                padding='same', activation='relu')(x)
        x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=filters[0], kernel_size=(3, 3),
                            padding='valid', activation='relu')(x)
    x = keras.layers.UpSampling2D(size=(2, 2))(x)

    x = keras.layers.Conv2D(filters=input_dims[-1], kernel_size=(3, 3),
                            padding='same', activation='sigmoid')(x)

    return keras.models.Model(decoder_input, x)


def autoencoder(input_dims, filters, latent_dims):
    """
    Creates a convolutional autoencoder.
    """
    encoder = build_encoder(input_dims, filters)
    decoder = build_decoder(latent_dims, filters, input_dims)

    encoder_input = keras.layers.Input(shape=input_dims)
    encoded_output = encoder(encoder_input)
    decoded_output = decoder(encoded_output)

    auto = keras.Model(inputs=encoder_input, outputs=decoded_output)
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
