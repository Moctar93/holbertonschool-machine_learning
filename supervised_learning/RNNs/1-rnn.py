#!/usr/bin/env python3
"""
RNN function performing forward propagation
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation.
    """
    t, m, i = X.shape
    _, h = h_0.shape

    H = np.zeros((t + 1, m, h))
    H[0] = h_0

    outputs = []

    for step in range(t):
        h_prev = H[step]
        x_t = X[step]
        h_next, y = rnn_cell.forward(h_prev, x_t)
        H[step + 1] = h_next
        outputs.append(y)

    Y = np.array(outputs)

    return H, Y
