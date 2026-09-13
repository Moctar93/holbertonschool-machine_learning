#!/usr/bin/env python3
"""
Module for flipping an image horizontally.
"""

import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally.

    Args:
        image: A 3D tf.Tensor containing an image.

    Returns:
        The horizontally flipped image.
    """
    return tf.image.flip_left_right(image)
