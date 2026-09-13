#!/usr/bin/env python3
import tensorflow as tf

"""
Module for changing the hue of an image.
"""


def change_hue(image, delta):
    """
    Changes the hue of an image.

    Args:
        image: A 3D tf.Tensor containing the image.
        delta: The amount by which to change the hue.

    Returns:
        The hue-adjusted image.
    """
    return tf.image.adjust_hue(image, delta)
