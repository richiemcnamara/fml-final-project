import numpy as np
import sys

class Node:
    def __init__(self, val, split_val, y_val=None):
        self.l = None
        self.r = None
        self.v = int(val) # -1 is a leaf
        self.split = split_val
        self.y = y_val


class PERTLearner:

    def __init__(self, leaf_size=1):
        # set up your object
        self.model = None
        self.leaf_size = leaf_size

    def train(self, x, y):
        # induce a decision tree based on this training data
        sys.setrecursionlimit(10**8)
        self.model = self.create_tree(x, y)


    def test(self, x):
        # return predictions (estimates) for each row of x
        predictions = []

        for i in range(len(x)):
            value = self.get_tree_value(x.iloc[i])
            predictions.append(value)
        return predictions

    
    def get_tree_value(self, x_i):
        current_node = self.model
        while current_node.y == None:
            feature = current_node.v
            split_value = current_node.split
            
            if x_i[feature] <= split_value:
                current_node = current_node.left
            else:
                current_node = current_node.right
        return current_node.y


    def create_tree(self, x, y):
        if x.shape[0] < self.leaf_size:
            return Node(-1, 0, y.mode().iloc[0])
        elif np.all(y == y[0]):
            return Node(-1, 0, y[0])
        elif np.all(x == x.iloc[0]):
            return Node(-1, 0, y.mode().iloc[0])
        else:
            fails = 0
            while True:
                # pick 2 random points until different y values
                if fails == 10:
                    # make leaf
                    return Node(-1, 0, y.mode().iloc[0])
                a_idx = np.random.randint(x.shape[0])
                b_idx = np.random.randint(x.shape[0])
                a = x.iloc[a_idx, :]
                b = x.iloc[b_idx, :]
                if y[a_idx] == y[b_idx]:
                    fails += 1
                else:
                    feature = np.random.randint(x.shape[1])
                    if a[feature] == b[feature]:
                        fails += 1
                    else:
                        break
            alpha = np.random.random()
            split_value = alpha * a[feature] + (1 - alpha) * b[feature]
                
            curr = Node(feature, split_value)

            curr.left = self.create_tree(x[x.iloc[:,feature] <= split_value], y[x.iloc[:,feature] <= split_value])
            curr.right = self.create_tree(x[x.iloc[:,feature] > split_value], y[x.iloc[:,feature] > split_value])

            return curr
