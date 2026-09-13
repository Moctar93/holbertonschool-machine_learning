#!/usr/bin/env python3
"""
Module for randomly adjusting image contrast.
"""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """
    Randomly adjusts the contrast of an image.

    Args:
        image: A 3D tf.Tensor representing an image.
        lower: Lower bound of the contrast factor.
        upper: Upper bound of the contrast factor.

    Returns:
        The contrast-adjusted image.
    """
    return tf.image.random_contrast(image, lower, upper)
