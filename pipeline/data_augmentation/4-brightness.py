#!/usr/bin/env python3
"""
Module for randomly changing image brightness.
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        max_delta: Maximum amount to brighten or darken the image.

    Returns:
        The brightness-adjusted image.
    """
    return tf.image.random_brightness(image, max_delta)
