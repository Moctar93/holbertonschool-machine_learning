#!/usr/bin/env python3
"""
Module for randomly cropping an image.
"""

import tensorflow as tf


def crop_image(image, size):
    """
    Randomly crops an image to the given size.

    Args:
        image: A 3D tf.Tensor containing an image.
        size: A tuple containing the size of the crop.

    Returns:
        The randomly cropped image.
    """
    return tf.image.random_crop(image, size)
