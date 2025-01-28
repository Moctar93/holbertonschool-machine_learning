#!/usr/bin/env python3

import numpy as np

"""
Module pour la création de la matrice de confusion.

Ce module contient une fonction qui génère une matrice de confusion
à partir des labels réels (en one-hot encoding) et des prédictions
du modèle (également en one-hot encoding).

La matrice de confusion permet d'évaluer la performance du modèle
en indiquant combien de fois chaque classe réelle a été correctement
ou incorrectement prédite.

La fonction 'create_confusion_matrix' prend deux entrées :
- 'labels' : un tableau de forme (m, classes) contenant les labels réels
- 'logits' : un tableau de forme (m, classes) contenant les prédictions

Elle retourne une matrice de confusion de forme (classes, classes),
avec chaque ligne représentant une classe réelle et chaque colonne
représentant une classe prédite.

"""


def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion à partir des labels réels et des
    logits prédits.

    Parameters:
    - labels : numpy.ndarray de forme (m, classes), les labels réels en
    one-hot encoding
    - logits : numpy.ndarray de forme (m, classes), les prédictions
    en one-hot encoding

    Returns:
    - numpy.ndarray de forme (classes, classes), la matrice de confusion
    """
    # Convertir les vecteurs one-hot en indices de classes
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    # Créer la matrice de confusion avec le type float
    confusion_matrix = np.zeros((labels.shape[1],)*2, dtype=float)

    # Remplir la matrice de confusion
    for true, pred in zip(true_classes, predicted_classes):
        confusion_matrix[true, pred] += 1.0

    return confusion_matrix
