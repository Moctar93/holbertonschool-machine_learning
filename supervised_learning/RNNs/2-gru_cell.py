#!/usr/bin/env python3
"""
GRU
"""

import numpy as np


class GRUCell:
    """
    GRU of a RNN
    """

    def __init__(self, i, h, o):
        """
        Initialize the GRU
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        forward propagation for one time step.
        """
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        z_t = self.sigmoid(np.dot(concat_h_x, self.Wz) + self.bz)

        r_t = self.sigmoid(np.dot(concat_h_x, self.Wr) + self.br)

        concat_r_h_x = np.concatenate((r_t * h_prev, x_t), axis=1)
        h_candidate = np.tanh(np.dot(concat_r_h_x, self.Wh) + self.bh)

        h_next = (1 - z_t) * h_prev + z_t * h_candidate

        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y

    @staticmethod
    def softmax(x):
        """
        Softmax method
        """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid method
        """
        return 1 / (1 + np.exp(-x))
