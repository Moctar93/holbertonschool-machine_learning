#!/usr/bin/env python3
"""
Transformer Decoder Block
"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    This class represents a transformer's decoder block.
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Initializes the decoder block.
        """
        super().__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Forward pass through the transformer's decoder block.
        """
        masked_mha_output, _ = self.mha1(x, x, x, look_ahead_mask)
        masked_mha_output = self.dropout1(masked_mha_output, training=training)
        output1 = self.layernorm1(x + masked_mha_output)

        mha2_output, _ = self.mha2(output1, encoder_output, encoder_output,
                                   padding_mask)
        mha2_output = self.dropout2(mha2_output)

        output2 = self.layernorm2(mha2_output + output1)

        ff_output = self.dense_hidden(output2)
        ff_output = self.dense_output(ff_output)
        ff_output = self.dropout3(ff_output, training=training)

        output2 = self.layernorm3(ff_output + output2)

        return output2
