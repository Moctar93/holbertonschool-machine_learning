#!/usr/bin/env python3
"""
Calculate the accuracy of the prediction.
"""

import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    Calculate the accuracy of a prediction.
    """
    y_pred_labels = tf.argmax(y_pred, axis=1)
    y_true_labels = tf.argmax(y, axis=1)

    correct_preds = tf.equal(y_pred_labels, y_true_labels)

    return tf.reduce_mean(tf.cast(correct_preds, tf.float32))
