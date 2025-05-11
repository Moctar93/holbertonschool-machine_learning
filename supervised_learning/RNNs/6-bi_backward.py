#!/usr/bin/env python3
"""
Bidirectional Cell
"""
import numpy as np


class BidirectionalCell:
    """
    bidirectional cell of a RNN
    """
    def __init__(self, i, h, o):
        """
        Initialize a bidirectional cell for an RNN
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h * 2, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step
        """
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        return np.tanh(np.dot(concat_h_x, self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """
        Perform backward propagation for one time step.
        """
        concat_h_x = np.concatenate((h_next, x_t), axis=1)

        return np.tanh(np.dot(concat_h_x, self.Whb) + self.bhb)
