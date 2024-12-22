#!/usr/bin/env python3

class Node:
    """Initialise un nœud interne."""
    def __init__(self, feature, threshold, left_child=None, right_child=None, depth=0, is_root=False):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth
        self.is_root = is_root
        self.upper = {}
        self.lower = {}

    """La méthode update_bounds_below est récursive"""
    def update_bounds_below(self):
        if self.is_root:
            self.upper = {self.feature: float('inf')}
            self.lower = {self.feature: float('-inf')}

        for child, direction in [(self.left_child, "left"), (self.right_child, "right")]:
            if child:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()

                if direction == "left":
                    child.upper[self.feature] = min(child.upper.get(self.feature, float('inf')), self.threshold)
                elif direction == "right":
                    child.lower[self.feature] = max(child.lower.get(self.feature, float('-inf')), self.threshold)

                child.update_bounds_below()

"""Un nœud Leaf ne possède pas d'enfants"""
class Leaf(Node):
    def __init__(self, value, depth=0):
        super().__init__(feature=None, threshold=None, depth=depth)
        self.value = value

    def update_bounds_below(self):
        pass  # No children to propagate bounds to in a leaf.

"""La méthode update_bounds commence par la racine de l'arbre"""
class Decision_Tree:
    def __init__(self, root=None):
        self.root = root

    def update_bounds(self):
        if self.root:
            self.root.update_bounds_below()

    def get_leaves(self):
        def collect_leaves(node):
            if isinstance(node, Leaf):
                return [node]
            leaves = []
            if node.left_child:
                leaves.extend(collect_leaves(node.left_child))
            if node.right_child:
                leaves.extend(collect_leaves(node.right_child))
            return leaves

        return collect_leaves(self.root)

