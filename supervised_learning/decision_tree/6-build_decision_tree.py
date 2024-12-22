#!/usr/bin/env python3

import numpy as np

class Leaf:
    def __init__(self, depth, value):
        self.depth = depth
        self.value = value

    def pred(self, x):
        return self.value

class Node:
    def __init__(self, depth, is_root=False):
        self.depth = depth
        self.is_root = is_root
        self.left_child = None
        self.right_child = None
        self.feature = None
        self.threshold = None
        self.lower = {}
        self.upper = {}

    def pred(self, x):
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)

class Decision_Tree:
    def __init__(self, root):
        self.root = root
        self.predict = None  # To be updated

    def update_bounds(self):
        # Implement the logic to update bounds
        pass

    def get_leaves(self):
        # Traverse the tree and collect all leaves
        leaves = []

        def traverse(node):
            if isinstance(node, Leaf):
                leaves.append(node)
            else:
                traverse(node.left_child)
                traverse(node.right_child)

        traverse(self.root)
        return leaves

    def update_predict(self):
        self.update_bounds()
        leaves = self.get_leaves()
        for leaf in leaves:
            leaf.update_indicator()

        self.predict = lambda A: [self.pred(x) for x in A]

    def pred(self, x):
        return self.root.pred(x)
