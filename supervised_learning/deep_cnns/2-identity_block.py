#!/usr/bin/env python3
"""
Identity Block
"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """
    Function that builds an identity block as described in Deep Residual Learning for Image Recognition (2015).

    Arguments:
    A_prev -- output from the previous layer
    filters -- tuple or list containing F11, F3, F12, respectively:
              F11 is the number of filters in the first 1x1 convolution
              F3 is the number of filters in the 3x3 convolution
              F12 is the number of filters in the second 1x1 convolution

    Returns:
    output -- activated output of the identity block
    """
    # Unpack the filters
    F11, F3, F12 = filters

    # Initializer
    initializer = K.initializers.HeNormal(seed=0)

    # First component of the main path
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=1,
                        strides=1,
                        padding='valid',
                        kernel_initializer=initializer)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Second component of the main path
    X = K.layers.Conv2D(filters=F3,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    # Third component of the main path
    X = K.layers.Conv2D(filters=F12,
                        kernel_size=1,
                        strides=1,
                        padding='valid',
                        kernel_initializer=initializer)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    # Add the input to the output (skip connection)
    X = K.layers.Add()([X, A_prev])

    # Final activation
    output = K.layers.Activation('relu')(X)

    return output
