#!/usr/bin/env python3
"""
Epsilon-greedy policy
"""

import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    epsilon-greedy policy for selecting the next action.
    """
    if np.random.uniform(0, 1) < epsilon:
        return np.random.randint(Q.shape[1])
    else:
        return np.argmax(Q[state])
