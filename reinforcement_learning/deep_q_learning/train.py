#!/usr/bin/env python3
"""
Training script
"""


from keras import __version__
import tensorflow as tf
tf.keras.__version__ = __version__

import cv2
import numpy as np

from rl.processors import Processor
from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Permute
from keras.optimizers.legacy import Adam
import gymnasium as gym
import matplotlib.pyplot as plt



class AtariProcessor(Processor):
    """Preprocessing Images"""

    def process_observation(self, observation):
        if isinstance(observation, tuple):
            observation = observation[0]
        # Ensure it's a NumPy array
        observation = np.array(observation)
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(img, (84, 84))
        return img

    def process_state_batch(self, batch):
        processed_batch = batch.astype('float32') / 255.
        return processed_batch

    def process_reward(self, reward):
        return np.clip(reward, -1., 1.)

class CompatibilityWrapper(gym.Wrapper):
    """
    Compatibility wrapper for gym env to ensure
    compatibility with older versions of gym
    """

    def step(self, action):
        """
        Take a step in the env using the given action
        """
        observation, reward, terminated, truncated, info = (
            self.env.step(action))
        done = terminated or truncated

        return observation, reward, done, info

    def reset(self, **kwargs):
        """
        Reset env and return the initial obs
        """
        observation, info = self.env.reset(**kwargs)

        return observation


def build_model(input_shape, actions):
    """
    """
    model = Sequential()
    model.add(Permute((2, 3, 1), input_shape=(4, 84, 84)))
    model.add(Conv2D(32, (8, 8), strides=4, activation='relu'))
    model.add(Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model


def build_agent(model, actions):
    """
    """
    memory = SequentialMemory(limit=100000, window_length=4)
    policy = LinearAnnealedPolicy(
        EpsGreedyQPolicy(),
        attr='eps',
        value_max=1.,
        value_min=.1,
        value_test=.05,
        nb_steps=4000
    )
    dqn = DQNAgent(model=model,
                   nb_actions=actions,
                   memory=memory,
                   nb_steps_warmup=50000,
                   target_model_update=1e-2,
                   processor=AtariProcessor(),
                   gamma=.99,
                   policy=policy,
                   train_interval=4)
    dqn.compile(Adam(learning_rate=0.00025), metrics=['mae'])
    return dqn


def main():
    """
    """
    env = gym.make("ALE/Breakout-v5")
    env = CompatibilityWrapper(env)
    observation = env.reset()

    plt.imshow(observation, cmap='gray')
    plt.title("Initial Observation")
    plt.axis('off')
    plt.show()

    actions = env.action_space.n
    model = build_model(observation.shape, actions)
    dqn = build_agent(model, actions)

    dqn.fit(env,
            nb_steps=100000,
            visualize=False,
            verbose=2)
    dqn.save_weights('policy.h5', overwrite=True)

    env.close()


if __name__ == "__main__":
    main()
