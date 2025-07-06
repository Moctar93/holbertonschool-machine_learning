#!/usr/bin/env python3
"""
Play the game by exploiting the Q-table
"""
import numpy as np


def play(env, Q, max_steps=100):
    """
    Simulates an episode of the environment.
    """
    state = env.reset()[0]

    done = False
    total_rewards = 0
    rendered_outputs = []

    for _ in range(max_steps):
        rendered_outputs.append(env.render())

        action = np.argmax(Q[state])

        next_state, reward, done, _, _ = env.step(action)

        total_rewards += reward

        state = next_state

        if done:
            break

    rendered_outputs.append(env.render())

    return total_rewards, rendered_outputs
