#!/usr/bin/env python3
"""
Long Short-Term Memory unit
"""

import numpy as np


class LSTMCell:
    """
    LSTM unit of a RNN
    """

    def __init__(self, i, h, o):
        """
        Initialize the LSTM unit
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)

        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        forward propagation for one time step.
        """
        concat_h_x = np.concatenate((h_prev, x_t), axis=1)

        f_t = self.sigmoid(np.dot(concat_h_x, self.Wf) + self.bf)

        u_t = self.sigmoid(np.dot(concat_h_x, self.Wu) + self.bu)

        o_t = self.sigmoid(np.dot(concat_h_x, self.Wo) + self.bo)

        c_act = np.tanh(np.dot(concat_h_x, self.Wc) + self.bc)

        c_next = f_t * c_prev + u_t * c_act

        h_next = o_t * np.tanh(c_next)

        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, c_next, y

    @staticmethod
    def softmax(x):
        """
        softmax method
        """
        return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    @staticmethod
    def sigmoid(x):
        """
        sigmoid method
        """
        return 1 / (1 + np.exp(-x))
