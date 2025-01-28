#!/usr/bin/env python3

import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion.

    Args:
        labels (numpy.ndarray): Un tableau one-hot de forme (m, classes) contenant les étiquettes correctes.
        logits (numpy.ndarray): Un tableau one-hot de forme (m, classes) contenant les prédictions.

    Returns:
        numpy.ndarray: Une matrice de confusion de forme (classes, classes).
    """
    # Convertir les one-hot vectors en indices de classe
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    # Nombre de classes
    num_classes = labels.shape[1]

    # Initialiser la matrice de confusion
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    # Remplir la matrice de confusion
    for true, predicted in zip(true_labels, predicted_labels):
        confusion_matrix[true, predicted] += 1

    return confusion_matrix
