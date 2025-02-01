#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2
"""

import tensorflow as tf


def l2_reg_cost(cost, model):
    """
    calculates the cost of a neural network with L2
    """
    return cost + model.losses
