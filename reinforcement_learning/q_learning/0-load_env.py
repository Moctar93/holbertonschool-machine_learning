#!/usr/bin/env python3
"""
loading a FrozenLake environment.
"""

import gymnasium as gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    Loading the pre-made FrozenLakeEnv env.
    """
    env = gym.make("FrozenLake-v1",
                   desc=desc,
                   map_name=map_name,
                   is_slippery=is_slippery,
                   render_mode="ansi")
    return env
