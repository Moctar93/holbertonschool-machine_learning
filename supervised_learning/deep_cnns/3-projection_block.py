#!/usr/bin/env python3
"""
Projection Block
"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """
    Function that builds a projection block as described in Deep Residual Learning for Image Recognition (2015).

    Arguments:
    A_prev -- output from the previous layer
    filters -- tuple or list containing F11, F3, F12, respectively:
              F11 is the number of filters in the first 1x1 convolution
              F3 is the number of filters in the 3x3 convolution
              F12 is the number of filters in the second 1x1 convolution as well as the 1x1 convolution in the shortcut connection
    s -- stride of the first convolution in both the main path and the shortcut connection (default is 2)

    Returns:
    output -- activated output of the projection block
    """
    # Unpack the filters
    F11, F3, F12 = filters

    # Initializer
    initializer = K.initializers.HeNormal(seed=0)

    # Main path
    # First component of the main path
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=1,
                        strides=s,
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

    # Shortcut path
    X_shortcut = K.layers.Conv2D(filters=F12,
                                 kernel_size=1,
                                 strides=s,
                                 padding='valid',
                                 kernel_initializer=initializer)(A_prev)
    X_shortcut = K.layers.BatchNormalization(axis=3)(X_shortcut)

    # Add the main path and the shortcut path
    X = K.layers.Add()([X, X_shortcut])

    # Final activation
    output = K.layers.Activation('relu')(X)

    return output
