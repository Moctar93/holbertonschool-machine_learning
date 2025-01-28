import numpy as np

def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion à partir des labels réels et des logits prédits.

    Parameters:
    - labels : numpy.ndarray de forme (m, classes), les labels réels en one-hot encoding
    - logits : numpy.ndarray de forme (m, classes), les prédictions en one-hot encoding

    Returns:
    - numpy.ndarray de forme (classes, classes), la matrice de confusion
    """
    # Convertir les vecteurs one-hot en indices de classes
    true_classes = np.argmax(labels, axis=1)
    predicted_classes = np.argmax(logits, axis=1)

    # Créer la matrice de confusion avec le type float
    confusion_matrix = np.zeros((labels.shape[1], labels.shape[1]), dtype=float)

    # Remplir la matrice de confusion
    for true, pred in zip(true_classes, predicted_classes):
        confusion_matrix[true, pred] += 1.0  # Assurez-vous que les valeurs sont flottantes

    return confusion_matrix
