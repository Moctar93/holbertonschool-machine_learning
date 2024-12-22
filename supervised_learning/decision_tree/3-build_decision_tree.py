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

    def count_nodes_below(self, only_leaves=False):
        """
        Count the number of nodes under this node.
        """
        if only_leaves and self.is_leaf:
            return 1

        if not self.is_leaf:
            return (
                self.left_child.count_nodes_below(only_leaves=only_leaves) +
                self.right_child.count_nodes_below(only_leaves=only_leaves) +
                (not only_leaves)
            )

    def get_leaves_below(self):
        """
        Returns a list of all leaves in the subtree rooted at this node.
        """
        if self.is_leaf:
            return [self]
        leaves = []
        if self.left_child:
            leaves.extend(self.left_child.get_leaves_below())
        if self.right_child:
            leaves.extend(self.right_child.get_leaves_below())
        return leaves

    def __str__(self):
        """
        String representation of the node.
        """
        node_type = "root" if self.is_root else "node"
        node_representation = (f"{node_type} [feature={self.feature}, "
                               f"threshold={self.threshold}]\n")
        if self.left_child:
            left_str = self.left_child.__str__().replace("\n", "\n    |  ")
            node_representation += f"    +---> {left_str}"

        if self.right_child:
            right_str = self.right_child.__str__().replace("\n", "\n       ")
            node_representation += f"\n    +---> {right_str}"

        return node_representation


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

    def count_nodes_below(self, only_leaves=False):
        """
        Overwrites the same method for the Node class.
        Returns 1.
        """
        return 1

    def get_leaves_below(self):
        """
        Returns the current leaf in a list since
        it is the only leaf below itself.
        """
        return [self]

    def __str__(self):
        return (f"-> leaf [value={self.value}]")


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

    def count_nodes(self, only_leaves=False):
        """
        Counts the total nodes or only leaf nodes in the tree
        """
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def get_leaves(self):
        """
        Returns the list of all leaves in the tree.
        """
        return self.root.get_leaves_below()

    def __str__(self):
        return f"{self.root.__str__()}\n"
