#!/usr/bin/env python3
"""
RNN encoder
"""

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    A class representing a RNN encoder.
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor of RNNEncoder
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden states
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Return the full sequence of output
        """
        x = self.embedding(x)

        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
