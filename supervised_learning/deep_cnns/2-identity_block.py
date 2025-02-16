#!/usr/bin/env python3
"""
Implementation of a residual block and a model using it.
"""

from tensorflow import keras as K

def residual_block(x, filters):
    """
    A residual block with two convolutional layers and a skip connection.
    """
    # Shortcut
    shortcut = x

    # First convolution
    x = K.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.ReLU()(x)

    # Second convolution
    x = K.layers.Conv2D(filters, (3, 3), padding='same')(x)
    x = K.layers.BatchNormalization()(x)

    # Add shortcut
    x = K.layers.Add()([x, shortcut])
    x = K.layers.ReLU()(x)

    return x

def build_model(input_shape=(56, 56, 256)):
    """
    Builds a model with residual blocks.
    """
    inputs = K.Input(shape=input_shape)

    # Initial convolution
    x = K.layers.Conv2D(64, (7, 7), padding='same')(inputs)
    x = K.layers.BatchNormalization()(x)
    x = K.layers.ReLU()(x)

    # Residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    # Final layers
    x = K.layers.GlobalAveragePooling2D()(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.models.Model(inputs, outputs)
    return model
