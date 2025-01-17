#!/usr/bin/env python3
"""
training operation for the network.
"""

import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """
    training operation for the network.
    """
    gradient_descent = tf.train.GradientDescentOptimizer(learning_rate=alpha)

    return gradient_descent.minimize(loss)
