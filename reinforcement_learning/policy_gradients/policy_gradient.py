#!/usr/bin/env python3
"""
Compute the Monte-Carlo policy gradient.
"""

import numpy as np


def policy(matrix, weight):
    """
    Computing the policy
    """
    z = np.dot(matrix, weight)
    exp = np.exp(z - np.max(z))
    return exp / exp.sum(axis=1, keepdims=True)


def policy_gradient(state, weight):
    """
    Computes the Monte-Carlo policy gradient based on a state and a
    weight matrix.
    """

    state = state.reshape(1, -1)

    probs = policy(state, weight).flatten()

    action = np.random.choice(len(probs), p=probs)

    dsoftmax = probs.copy()
    dsoftmax[action] -= 1
    gradient = np.outer(state, -dsoftmax)

    return action, gradient
