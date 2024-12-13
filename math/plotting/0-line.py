#!/usr/bin/env python3
import matplotlib
matplotlib.use('TkAgg')  # Set an interactive backend
import numpy as np
import matplotlib.pyplot as plt

def line():
    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))
    x = np.arange(0, 11)
    plt.plot(x, y, 'r-')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of y = x^3')
    plt.show()

line()

