#!/usr/bin/env python3
"""
Convert Gensim Word2Vec.
"""

import tensorflow as tf


def gensim_to_keras(model):
    """
    Converting a trained gensim Word2vec model
    to a Keras Embedding layer.
    """
    weights = model.wv.vectors
    embedding_layer = tf.keras.layers.Embedding(
            input_dim=weights.shape[0],
            output_dim=weights.shape[1],
            weights=[weights],
            trainable=True
            )
    return embedding_layer
