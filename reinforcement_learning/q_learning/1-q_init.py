#!/usr/bin/env python3
"""
Initialize Q-table
"""

import numpy as np


def q_init(env):
    """
    Initialize Q-table
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    np.zeros((num_states, num_actions))
    return np.zeros((num_states, num_actions))
