#!/usr/bin/env python3
"""
Create a tensor layer.
"""

import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """
    Create a layer for a neural network.
    """
    init_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.keras.layers.Dense(units=n, activation=activation,
                                  kernel_initializer=init_weights,
                                  name="layer")

    return layer(prev)
