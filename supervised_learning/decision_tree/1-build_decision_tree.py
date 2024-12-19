class Node:
    def __init__(self, feature, threshold, left_child, right_child, depth, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.is_leaf = False

    def count_nodes_below(self, only_leaves=False):
        """
        Compte les noeuds sous ce noeud (y compris lui-même) ou uniquement les feuilles si only_leaves=True.
        """
        if only_leaves:
            # Compter uniquement les feuilles
            return self.left_child.count_nodes_below(only_leaves=True) + \
                   self.right_child.count_nodes_below(only_leaves=True)
        else:
            # Compter tous les noeuds
            return 1 + self.left_child.count_nodes_below() + self.right_child.count_nodes_below()


class Leaf:
    def __init__(self, value, depth):
        self.value = value
        self.depth = depth
        self.is_leaf = True

    def count_nodes_below(self, only_leaves=False):
        """
        Retourne 1 si c'est une feuille, car une feuille est comptée comme un noeud terminal.
        """
        return 1


class Decision_Tree:
    def __init__(self, root):
        self.root = root

    def count_nodes(self, only_leaves=False):
        """
        Compte tous les noeuds dans l'arbre ou uniquement les feuilles si only_leaves=True.
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

