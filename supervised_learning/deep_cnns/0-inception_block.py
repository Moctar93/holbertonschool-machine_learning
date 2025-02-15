#!/usr/bin/env python3
"""
Inception Block
"""
from tensorflow import keras as K


def inception_block(A_prev, filters):
    """
    Function that builds an inception block as described in Going Deeper
    with Convolutions (2014).

    Arguments:
    A_prev -- input tensor from the previous layer
    filters -- tuple or list of 6 integers, specifying the number
    of filters in each convolution:
               [F1, F3R, F3, F5R, F5, FPP]

    Returns:
    output -- concatenated output of the inception block
    """
    # Initializer
    initializer = K.initializers.HeNormal()

    # 1x1 Convolution Branch
    F1 = K.layers.Conv2D(filters=filters[0],
                         kernel_size=1,
                         padding='same',
                         kernel_initializer=initializer,
                         activation='relu')(A_prev)

    # 3x3 Convolution Branch
    F3R = K.layers.Conv2D(filters=filters[1],
                          kernel_size=1,
                          padding='same',
                          kernel_initializer=initializer,
                          activation='relu')(A_prev)
    F3 = K.layers.Conv2D(filters=filters[2],
                         kernel_size=3,
                         padding='same',
                         kernel_initializer=initializer,
                         activation='relu')(F3R)

    # 5x5 Convolution Branch
    F5R = K.layers.Conv2D(filters=filters[3],
                          kernel_size=1,
                          padding='same',
                          kernel_initializer=initializer,
                          activation='relu')(A_prev)
    F5 = K.layers.Conv2D(filters=filters[4],
                         kernel_size=5,
                         padding='same',
                         kernel_initializer=initializer,
                         activation='relu')(F5R)

    # Max-Pooling Branch
    Pool = K.layers.MaxPool2D(pool_size=3,
                              strides=1,
                              padding='same')(A_prev)
    FPP = K.layers.Conv2D(filters=filters[5],
                          kernel_size=1,
                          padding='same',
                          kernel_initializer=initializer,
                          activation='relu')(Pool)

    # Concatenate all branches
    output = K.layers.concatenate([F1, F3, F5, FPP])

    return output
