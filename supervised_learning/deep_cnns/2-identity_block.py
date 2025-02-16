#!/usr/bin/env python3
"""
Building an identity block
"""

from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Building an identity block
    """
    F11, F3, F12 = filters

    init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(A_prev)

    norm1 = K.layers.BatchNormalization(axis=-1)(conv1)
    relu1 = K.layers.Activation(activation="relu")(norm1)

    conv2 = K.layers.Conv2D(filters=F3,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu1)
    norm2 = K.layers.BatchNormalization(axis=-1)(conv2)
    relu2 = K.layers.Activation(activation="relu")(norm2)

    conv3 = K.layers.Conv2D(filters=F12,
                            kernel_size=(1, 1),
                            strides=(1, 1),
                            padding="same",
                            kernel_initializer=init)(relu2)
    norm3 = K.layers.BatchNormalization(axis=-1)(conv3)

    merged = K.layers.Add()([norm3, A_prev])

    return K.layers.Activation(activation="relu")(merged)
