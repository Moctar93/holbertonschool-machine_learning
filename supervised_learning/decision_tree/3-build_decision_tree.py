#!/usr/bin/env python3

class Node:
    """
    Représente un nœud interne dans un arbre de décision.

    Attributs :
    ----------
    - feature : int
        L'indice de la caractéristique utilisée pour diviser.
    - threshold : float
        Le seuil utilisé pour diviser les données.
    - left_child : Node ou Leaf
        Le sous-arbre gauche ou feuille associé à ce nœud.
    - right_child : Node ou Leaf
        Le sous-arbre droit ou feuille associé à ce nœud.
    - is_root : bool
        Indique si ce nœud est la racine de l'arbre.
    - depth : int
        La profondeur du nœud dans l'arbre.
    - is_leaf : bool
        Toujours `False` pour un nœud interne (contrairement à une feuille).
    """

    def __init__(self, feature=None, threshold=None, left_child=None,
                 right_child=None, is_root=False, depth=0):
        """
        Initialise un nœud interne.

        Arguments :
        ----------
        - feature : int, optionnel
            L'indice de la caractéristique utilisée pour diviser.
        - threshold : float, optionnel
            Le seuil utilisé pour diviser les données.
        - left_child : Node ou Leaf, optionnel
            Le sous-arbre gauche ou feuille associé à ce nœud.
        - right_child : Node ou Leaf, optionnel
            Le sous-arbre droit ou feuille associé à ce nœud.
        - is_root : bool, optionnel
            Indique si ce nœud est la racine de l'arbre. Par défaut, `False`.
        - depth : int, optionnel
            La profondeur du nœud dans l'arbre. Par défaut, `0`.
        """
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False  # Toujours False pour un nœud interne

    def get_leaves_below(self):
        """
        Récupère toutes les feuilles sous ce nœud.

        Retourne :
        --------
        - list : Une liste contenant toutes les feuilles situées sous ce nœud.
        """
        leaves = []
        # Vérifie si un enfant gauche existe, puis explore récursivement
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        # Vérifie si un enfant droit existe, puis explore récursivement
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf:
    """
    Représente une feuille dans un arbre de décision.

    Attributs :
    ----------
    - value : int ou float
        La valeur assignée à la feuille.
    - depth : int
        La profondeur de la feuille dans l'arbre.
    - is_leaf : bool
        Toujours `True` pour une feuille.
    """

    def __init__(self, value, depth=0):
        """
        Initialise une feuille.

        Arguments :
        ----------
        - value : int ou float
            La valeur assignée à la feuille.
        - depth : int, optionnel
            La profondeur de la feuille dans l'arbre. Par défaut, `0`.
        """
        self.value = value
        self.depth = depth
        self.is_leaf = True  # Toujours True pour une feuille

    def get_leaves_below(self):
        """
        Retourne cette feuille, car elle est elle-même une feuille.

        Retourne :
        --------
        - list : Une liste contenant uniquement cette feuille.
        """
        return [self]

    def __repr__(self):
        """
        Fournit une représentation textuelle d'une feuille pour un affichage clair.

        Exemple :
        --------
        -> leaf [value=5]
        """
        return f"-> leaf [value={self.value}]"


class Decision_Tree:
    """
    Représente un arbre de décision avec un nœud racine.

    Attributs :
    ----------
    - root : Node
        Le nœud racine de l'arbre de décision.
    """

    def __init__(self, root):
        """
        Initialise l'arbre de décision.

        Arguments :
        ----------
        - root : Node
            Le nœud racine de l'arbre de décision.
        """
        self.root = root

    def get_leaves(self):
        """
        Récupère toutes les feuilles de l'arbre de décision.

        Retourne :
        --------
        - list : Une liste contenant toutes les feuilles de l'arbre.
        """
        return self.root.get_leaves_below()
