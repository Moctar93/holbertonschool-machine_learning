#!/usr/bin/env python3
"""
performing a same convolution on grayscale images.
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performing a same convolution on grayscale images.
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    convolved_output = np.zeros((m, h, w))

    pad_width = ((0, 0), (kh // 2, kh // 2), (kw // 2, kw // 2))
    padded_images = np.pad(
            images, pad_width=pad_width, mode="constant", constant_values=0
            )

    for i in range(h):
        for j in range(w):
            region = padded_images[:, i:(i + kh), j:(j + kw)]
            convolved_output[:, i, j] = np.sum(region * kernel, axis=(1, 2))

    return convolved_output
