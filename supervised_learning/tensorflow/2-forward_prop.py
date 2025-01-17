#!/usr/bin/env python3
"""
Module to perform forward propagation
"""
import tensorflow.compat.v1 as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    Creates the forward propagation graph for the neural network.
    
    :param x: tf.placeholder, the input data placeholder
    :param layer_sizes: list, containing the number of nodes in each layer
    :param activations: list, containing the activation functions for each layer
    :return: the prediction of the network in tensor form
    """
    prediction = x  # Start with the input placeholder
    for i in range(len(layer_sizes)):
        prediction = create_layer(prediction, layer_sizes[i], activations[i])
    return prediction

