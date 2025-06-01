#!/usr/bin/env python3
"""
Positional Encoding
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer.
    """
    pos_encoding_vectors = np.zeros(shape=(max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm // 2):
            div_term = 10000 ** (2 * i / dm)

            pos_encoding_vectors[pos, 2*i] = np.sin(pos / div_term)

            pos_encoding_vectors[pos, 2*i + 1] = np.cos(pos / div_term)

    return pos_encoding_vectors
