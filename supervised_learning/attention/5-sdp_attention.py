#!/usr/bin/env python3
"""
Scaled dot product Attention
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    Calculates the scaled dot product attention.
    """
    dk = tf.cast(Q.shape[-1], dtype=tf.float32)

    scores = tf.matmul(Q, K, transpose_b=True)

    scaled_scores = scores / tf.sqrt(dk)

    if mask:
        scaled_scores += (mask * -1e-9)

    attention_weights = tf.nn.softmax(scaled_scores, axis=-1)

    output = tf.matmul(attention_weights, V)

    return output, attention_weights
