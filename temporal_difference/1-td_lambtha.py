#!/usr/bin/env python3
"""
TD lambtha algorithm
"""
import numpy as np


def td_lambtha(env, V, policy, lambtha=0.9, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    TD(λ) algorithm for estimating the value function.
    """
    for episode in range(episodes):
        state = env.reset()[0]

        eligibility_traces = np.zeros_like(V)

        for step in range(max_steps):
            action = policy(state)

            next_state, reward, terminated, truncated, _ = env.step(action)

            delta = reward + (gamma * V[next_state] - V[state])

            eligibility_traces[state] += 1

            V += alpha * delta * eligibility_traces

            eligibility_traces *= gamma * lambtha

            state = next_state

            if terminated or truncated:
                break

    return V
