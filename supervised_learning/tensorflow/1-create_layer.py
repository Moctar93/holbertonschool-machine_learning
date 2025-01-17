#!/usr/bin/env python3
"""
Module to create a layer
"""
import tensorflow as tf

def create_layer(prev, n, activation):
    """
    Creates a layer using TensorFlow 2.x
    :param prev: tensor containing the output of the previous layer
    :param n: number of nodes in the layer
    :param activation: activation function to use
    :return: the output of the new layer
    """
    # Initialisation avec VarianceScaling (remplace tf.contrib.layers.variance_scaling_initializer)
    init = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_avg', distribution='uniform')
    
    # Création d'une couche Dense
    layer = tf.keras.layers.Dense(units=n, activation=activation, kernel_initializer=init, name="layer")
    return layer(prev)
