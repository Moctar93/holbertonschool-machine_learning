#!/usr/bin/env python3
"""
Deep RNN
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    forward propagation for a deep RNN.
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape

    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    H[0] = h_0

    for step in range(t):
        for layer, cell in enumerate(rnn_cells):
            if layer == 0:
                h_next, y = cell.forward(H[step, layer], X[step])
            else:
                h_next, y = cell.forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y[step] = y

    return H, Y
