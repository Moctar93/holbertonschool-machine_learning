#!/usr/bin/env python3
"""
softmax cross-entropy loss of a prediction.
"""

import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    the softmax cross-entropy loss of a prediction.
    """
    return tf.losses.softmax_cross_entropy(onehot_labels=y,
                                           logits=y_pred)
