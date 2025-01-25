import numpy as np
import sys

class Node:
    def __init__(self, val, split_val, y_val=None):
        self.l = None
        self.r = None
        self.v = int(val) # -1 is a leaf
        self.split = split_val
        self.y = y_val


class CARTLearner:

    def __init__(self, leaf_size=1):
        # set up your object
        self.tree = None
        self.leaf_size = leaf_size

    def train(self, x, y):
        # induce a decision tree based on this training data
        sys.setrecursionlimit(10**8)
        self.tree = self.create_tree(x, y)


    def test(self, x):
        # return predictions (estimates) for each row of x
        predictions = []

        for i in range(len(x)):
            value = self.get_tree_value(x.iloc[i])
            predictions.append(value)
        return predictions

    
    def get_tree_value(self, x_i):
        current_node = self.tree
        while current_node.y == None:
            feature = current_node.v
            split_value = current_node.split
            
            if x_i[feature] <= split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.y


    def create_tree(self, x, y):
        #     return this decision node
        if x.shape[0] < self.leaf_size:
            return Node(-1, 0, y.mode().iloc[0])
        elif np.all(y == y[0]):
            return Node(-1, 0, y[0])
        elif np.all(x == x.iloc[0]):
            return Node(-1, 0, y.mode().iloc[0])
        else:
            feature = self.get_best_feature(x, y)
            split_value = np.median(x.iloc[:,feature])
            max_value = max(x.iloc[:,feature])

            if split_value == max_value:
                return Node(-1, 0, y.mode().iloc[0])
                
            curr = Node(feature, split_value)

            curr.left = self.create_tree(x[x.iloc[:,feature] <= split_value], y[x.iloc[:,feature] <= split_value])
            curr.right = self.create_tree(x[x.iloc[:,feature] > split_value], y[x.iloc[:,feature] > split_value])

            return curr


    def get_best_feature(self, x, y):
        max_value = 0
        best_feature = 0

        for i in range(len(x.iloc[0])):
            np.seterr(invalid='ignore')
            correlation_value = abs(np.corrcoef(x.iloc[:,i],y)[0,1])
            if correlation_value > max_value:
                max_value = correlation_value
                best_feature = i
        return best_feature
        