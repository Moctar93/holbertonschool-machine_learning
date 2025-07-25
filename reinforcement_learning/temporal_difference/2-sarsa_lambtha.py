#!/usr/bin/env python3
"""
SARSA(λ) algorithm
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Determines the next action using the epsilon-greedy policy.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(Q[state, :])
    else:
        return np.random.randint(0, Q.shape[1])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm.
    """
    initial_epsilon = epsilon

    for episode in range(episodes):
        state = env.reset()[0]
        action = epsilon_greedy(Q, state, epsilon)

        eligibility_traces = np.zeros_like(Q)

        for steps in range(max_steps):
            new_state, reward, terminated, truncated, _ = env.step(action)

            new_action = epsilon_greedy(Q, new_state, epsilon)

            delta = (reward + (gamma * Q[new_state, new_action]) -
                     Q[state, action])

            eligibility_traces[state, action] += 1
            eligibility_traces *= lambtha * gamma

            Q += alpha * delta * eligibility_traces

            state = new_state
            action = new_action

            if terminated or truncated:
                break

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

    return Q
