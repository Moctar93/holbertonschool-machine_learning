#!/usr/bin/env python3
"""
Decision Tree
"""
import numpy as np


class Node:
    """
    Represents a node in the decision tree.
    """

    def __init__(
        self, feature=None, threshold=None, left_child=None,
        right_child=None, is_root=False, depth=0
    ):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """
        Computes the maximum depth of the subtree rooted at this node
        """

        max_depth = self.depth
        if self.left_child is not None:
            max_depth = max(max_depth, self.left_child.max_depth_below())

        if self.right_child is not None:
            max_depth = max(max_depth, self.right_child.max_depth_below())

        return max_depth


class Leaf(Node):
    """
    Represents a leaf in the decision tree.
    """

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """
        Returns the depth of the leaf since it has no children.
        """
        return self.depth


class Decision_Tree():
    """
    Represents the full decision tree model.
    """

    def __init__(
        self, max_depth=10, min_pop=1, seed=0,
        split_criterion="random", root=None
    ):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """
         Returns the maximum depth of the entire tree.
        """
        return self.root.max_depth_below()
