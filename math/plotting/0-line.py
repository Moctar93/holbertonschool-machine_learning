#!/usr/bin/env python3
"""
This module contains a script to plot y = x^3 as a solid red line
for x values ranging from 0 to 10 using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt


def line():
    """
    Function that generates a plot and
    displays it as a red solid line
    """

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    plt.plot(y, 'r-')
    plt.xlim(0, 10)
    plt.show()
