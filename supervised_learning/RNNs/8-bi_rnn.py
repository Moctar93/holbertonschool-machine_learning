#!/usr/bin/env python3
"""
Bidirectional RNN
"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Forward propagation for a bidirectional RNN.
    """
    t, m, _ = X.shape
    _, h = h_0.shape

    H = np.zeros((t, m, h * 2))
    Y = np.zeros((t, m, bi_cell.Wy.shape[1]))

    h_f = h_0
    h_b = h_t

    for step in range(t):
        h_f = bi_cell.forward(h_f, X[step])
        H[step, :, :h] = h_f

        h_b = bi_cell.backward(h_b, X[-1 - step])
        H[-1 - step, :, h:] = h_b

    Y = bi_cell.output(H)

    return H, Y
