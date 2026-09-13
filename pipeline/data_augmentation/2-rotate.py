#!/usr/bin/env python3
"""
Module for rotating an image 90 degrees counter-clockwise.
"""

import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise.

    Args:
        image: A 3D tf.Tensor containing an image.

    Returns:
        The rotated image.
    """
    return tf.image.rot90(image, k=1)
