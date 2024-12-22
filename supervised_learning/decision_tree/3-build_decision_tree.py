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
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_root = is_root
        self.depth = depth
        self.is_leaf = False

    def get_leaves_below(self):
        """Returns the list of all leaves of the tree below this node."""
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves


class Leaf:
    def __init__(self, value, depth=0):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def get_leaves_below(self):
        """Returns the list of all leaves (itself, as it is a leaf)."""
        return [self]

    def __repr__(self):
        return f"-> leaf [value={self.value}]"

class Decision_Tree:
    def __init__(self, root):
        self.root = root

    def get_leaves(self):
        """Returns the list of all leaves of the decision tree."""
        return self.root.get_leaves_below()
