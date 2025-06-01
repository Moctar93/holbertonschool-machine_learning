#!/usr/bin/env python3
"""
Self Attention
"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Self-attention mechanism for machine translation
    """

    def __init__(self, units):
        """
        Initializing SelfAttention layer.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Forward pass to compute attention mechanism.
        """
        s_prev_expanded = tf.expand_dims(input=s_prev, axis=1)

        scores = self.V(
            tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(scores, axis=1)

        context = tf.reduce_sum(weights * hidden_states, axis=1)

        return context, weights
