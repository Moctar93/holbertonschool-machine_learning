#!/usr/bin/env python3
"""
Building a dense block.
"""

from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Building a dense block.
    """
    init = K.initializers.HeNormal(seed=0)

    for layer_i in range(layers):
        norm1 = K.layers.BatchNormalization()(X)
        activ1 = K.layers.Activation(activation="relu")(norm1)

        conv1 = K.layers.Conv2D(filters=4 * growth_rate,
                                kernel_size=(1, 1),
                                padding="same",
                                kernel_initializer=init)(activ1)

        norm2 = K.layers.BatchNormalization()(conv1)
        activ2 = K.layers.Activation("relu")(norm2)
        conv2 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding="same",
                                kernel_initializer=init)(activ2)

        X = K.layers.Concatenate()([X, conv2])
        nb_filters += growth_rate

    return X, nb_filters
