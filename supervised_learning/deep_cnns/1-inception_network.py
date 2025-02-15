#!/usr/bin/env python3
"""
Inception Network
"""
from tensorflow import keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds the inception network as described in Going Deeper with Convolutions (2014).

    Returns:
    model -- the Keras model
    """
    # Input layer
    inputs = K.Input(shape=(224, 224, 3))

    # Initial layers
    x = K.layers.Conv2D(filters=64,
                        kernel_size=7,
                        strides=2,
                        padding='same',
                        activation='relu')(inputs)
    x = K.layers.MaxPool2D(pool_size=3,
                           strides=2,
                           padding='same')(x)
    x = K.layers.Conv2D(filters=64,
                        kernel_size=1,
                        strides=1,
                        padding='same',
                        activation='relu')(x)
    x = K.layers.Conv2D(filters=192,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu')(x)
    x = K.layers.MaxPool2D(pool_size=3,
                           strides=2,
                           padding='same')(x)

    # Inception blocks
    x = inception_block(x, [64, 96, 128, 16, 32, 32])  # Inception 3a
    x = inception_block(x, [128, 128, 192, 32, 96, 64])  # Inception 3b
    x = K.layers.MaxPool2D(pool_size=3,
                           strides=2,
                           padding='same')(x)
    x = inception_block(x, [192, 96, 208, 16, 48, 64])  # Inception 4a
    x = inception_block(x, [160, 112, 224, 24, 64, 64])  # Inception 4b
    x = inception_block(x, [128, 128, 256, 24, 64, 64])  # Inception 4c
    x = inception_block(x, [112, 144, 288, 32, 64, 64])  # Inception 4d
    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # Inception 4e
    x = K.layers.MaxPool2D(pool_size=3,
                           strides=2,
                           padding='same')(x)
    x = inception_block(x, [256, 160, 320, 32, 128, 128])  # Inception 5a
    x = inception_block(x, [384, 192, 384, 48, 128, 128])  # Inception 5b

    # Final layers
    x = K.layers.AveragePooling2D(pool_size=7,
                                  strides=1,
                                  padding='valid')(x)
    x = K.layers.Dropout(rate=0.4)(x)
    x = K.layers.Dense(units=1000,
                       activation='softmax')(x)

    # Create the model
    model = K.Model(inputs=inputs, outputs=x)

    return model
