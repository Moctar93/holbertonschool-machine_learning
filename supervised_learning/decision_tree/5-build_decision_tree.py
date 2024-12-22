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
        self.lower = {}
        self.upper = {}

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

    def update_bounds_below(self):
        """
        dictionaries contain the bounds for each feature.
        """
        if self.is_root:
            self.lower = {0: -np.inf}
            self.upper = {0: np.inf}

        if self.left_child:
            self.left_child.lower = self.lower.copy()
            self.left_child.upper = self.upper.copy()

            if self.feature in self.left_child.lower:
                self.left_child.lower[self.feature] = max(
                        self.threshold, self.left_child.lower[self.feature]
                        )
            else:
                self.left_child.lower[self.feature] = self.threshold

            self.left_child.update_bounds_below()

        if self.right_child:
            self.right_child.lower = self.lower.copy()
            self.right_child.upper = self.upper.copy()

            if self.feature in self.right_child.upper:
                self.right_child.upper[self.feature] = min(
                        self.threshold, self.right_child.upper[self.feature]
                        )
            else:
                self.right_child.upper[self.feature] = self.threshold

            self.right_child.update_bounds_below()

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

def update_indicator(self):
        """
        Update the indicator function based on the lower and upper bounds.
        """
        def is_large_enough(x):
            return np.all(
                np.array([np.greater(x[:, key], self.lower[key])
                          for key in self.lower]), axis=0
            )

        def is_small_enough(x):
            return np.all(
                np.array([np.less_equal(x[:, key], self.upper[key])
                          for key in self.upper]), axis=0
            )

        self.indicator = lambda x: np.all(
            np.array([is_large_enough(x), is_small_enough(x)]), axis=0)


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

    def update_bounds_below(self):
        """
        update the bounds for the node and its children.
        """
        pass

    def __str__(self):
        return (f"-> leaf [value={self.value}]")

    def update_indicator(self):
        """
        Updates the indicator function for this leaf node.
        """
        self.indicator = lambda x: np.ones(x.shape[0], dtype=bool) 


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

    def update_bounds(self):
        """
        Updates the lower and upper bounds for all nodes.
        """
        self.root.update_bounds_below()

    def __str__(self):
        return f"{self.root.__str__()}\n"
