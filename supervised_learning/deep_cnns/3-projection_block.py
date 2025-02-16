#!/usr/bin/env python3
"""
Building a projection Block.
"""

from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Building a projection block.
    """
    F11, F3, F12 = filters

    init = K.initializers.HeNormal(seed=0)

    conv1 = K.layers.Conv2D(filters=F11,
                            kernel_size=(1, 1),
                            strides=(s, s),
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

    conv_shortcut = K.layers.Conv2D(filters=F12,
                                    kernel_size=(1, 1),
                                    strides=(s, s),
                                    padding="same",
                                    kernel_initializer=init)(A_prev)
    norm_shortcut = K.layers.BatchNormalization(axis=-1)(conv_shortcut)

    merged = K.layers.Add()([norm3, norm_shortcut])

    return K.layers.Activation(activation="relu")(merged)
