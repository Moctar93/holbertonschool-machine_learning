#!/usr/bin/env python3
"""
Initialize Gaussian Process
"""

import numpy as np


class GaussianProcess:
    """
    This class represents a noiseless 1D Gaussian process.
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Initializes the Gaussian Process.
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        Calculates the RBF kernel 
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)

    def predict(self, X_s):
        """
        Predicts the mean and standard deviation.
        """
        K_s = self.kernel(X_s, self.X)
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(self.K)

        mu_s = K_s.dot(K_inv).dot(self.Y).flatten()

        sigma_s = K_ss - K_s.dot(K_inv).dot(K_s.T)

        sigma_s_diag = np.diag(sigma_s)

        return mu_s, sigma_s_diag
