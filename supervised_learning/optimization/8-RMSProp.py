#!/usr/bin/env python3
"""
Updates a variable using the RMSprop optimization algorithm with TensorFlow.
"""
import tensorflow as tf

def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    A function that optimizes using RMSprop with TensorFlow.
    :param loss: the loss of the network
    :param alpha: the learning rate
    :param beta2: the RMSprop weight
    :param epsilon: small number to avoid division by zero
    :return: the RMSprop optimization operation
    """
    # Create the RMSProp optimizer
    rms = tf.optimizers.RMSprop(learning_rate=alpha, rho=beta2, epsilon=epsilon)
    
    # Perform the optimization and minimize the loss
    return rms.minimize(loss)
