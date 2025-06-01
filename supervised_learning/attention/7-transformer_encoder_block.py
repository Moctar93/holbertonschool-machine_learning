#!/usr/bin/env python3
"""
Transformer Encoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    class represents a transformer's encoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the encoder block.
        """
        super().__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """
        Forward pass through the encoder block.
        """
        mha_output, _ = self.mha(x, x, x, mask)
        mha_output = self.dropout1(mha_output, training=training)
        output1 = self.layernorm1(x + mha_output)

        ff_output = self.dense_hidden(output1)
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout2(ff_output, training=training)

        output2 = self.layernorm2(output1 + ff_output)

        return output2
