import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5: 0.45,
             0.25: 1.32,
             0.1: 2.71,
             0.05: 3.84,
             0.0001: 100000},
             2: {0.5: 1.39,
             0.25: 2.77,
             0.1: 4.60,
             0.05: 5.99,
             0.0001: 100000},
             3: {0.5: 2.37,
             0.25: 4.11,
             0.1: 6.25,
             0.05: 7.82,
             0.0001: 100000},
             4: {0.5: 3.36,
             0.25: 5.38,
             0.1: 7.78,
             0.05: 9.49,
             0.0001: 100000},
             5: {0.5: 4.35,
             0.25: 6.63,
             0.1: 9.24,
             0.05: 11.07,
             0.0001: 100000},
             6: {0.5: 5.35,
             0.25: 7.84,
             0.1: 10.64,
             0.05: 12.59,
             0.0001: 100000},
             7: {0.5: 6.35,
             0.25: 9.04,
             0.1: 12.01,
             0.05: 14.07,
             0.0001: 100000},
             8: {0.5: 7.34,
             0.25: 10.22,
             0.1: 13.36,
             0.05: 15.51,
             0.0001: 100000},
             9: {0.5: 8.34,
             0.25: 11.39,
             0.1: 14.68,
             0.05: 16.92,
             0.0001: 100000},
             10: {0.5: 9.34,
                  0.25: 12.55,
                  0.1: 15.99,
                  0.05: 18.31,
                  0.0001: 100000},
             11: {0.5: 10.34,
                  0.25: 13.7,
                  0.1: 17.27,
                  0.05: 19.68,
                  0.0001: 100000}}


def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    square = np.square(np.unique(data[:, -1], return_counts=True)[1]) # uses np.square, sum to use vercorized version
    gini = 1 - np.sum(square)/(len(data)*len(data))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return gini


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pi = np.unique(data[:, -1], return_counts=True)[1] # uses np.log, dot, sum to use vercorized version
    pi = pi/len(data)
    logz = np.log(pi)
    entropy = -np.dot(logz, pi)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return entropy

# helper function that helps to calculate the gain ratio by returning the split info of some data after split by some feature

def split_information(data, feature):
    entropy = 0.0
    reg = np.unique(data[:, feature], return_counts=True)[1]
    reg = reg/len(data)
    logz = np.log2(reg)
    entropy = abs(-np.dot(logz, reg))
    return entropy


def goodness_of_split(data, feature, impurity_func, gain_ratio=False):
    """
    Calculate the goodness of split of a dataset given a feature and impurity function.
    Note: Python support passing a function as arguments to another function
    Input:
    - data: any dataset where the last column holds the labels.
    - feature: the feature index the split is being evaluated according to.
    - impurity_func: a function that calculates the impurity.
    - gain_ratio: goodness of split or gain ratio flag.

    Returns:
    - goodness: the goodness of split value
    - groups: a dictionary holding the data after splitting 
              according to the feature values.
    """
    goodness = 0
    groups = {}  # groups[feature_value] = data_subset
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    goodness = impurity_func(data)
    size = len(data)
    attribute_values = np.unique(data[:, feature])
    for value in attribute_values: # for each feature, finds the impurity by that feature
        groups[value] = data[data[:, feature] == value]
        size_group = len(groups[value])
        impurity_group = impurity_func(groups[value])
        goodness -= (size_group/size)*impurity_group # goodness is updated with relation to the relative size of the group
    if (gain_ratio): # normalization by split_info if required
        goodness = goodness/split_information(data, feature)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return goodness, groups


class DecisionNode:

    def __init__(self, data, feature=-1, depth=0, chi=1, max_depth=1000, gain_ratio=False):

        self.data = data  # the relevant data for the node
        self.feature = feature  # column index of criteria being tested
        self.pred = self.calc_node_pred()  # the prediction of the node
        self.depth = depth  # the current depth of the node
        self.children = []  # array that holds this nodes children
        self.children_values = []
        self.terminal = False  # determines if the node is a leaf
        self.parent = None # parent pointer field. added for helpful operations
        self.chi = chi
        self.max_depth = max_depth  # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio

    def calc_node_pred(self):
        # returns the most probable *value* of the label data of some node.

        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        u, c = np.unique(self.data[:, -1], return_counts=True) # np.unique counts unique values
        pred = u[c == c.max()][0] # returns the respective value

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def calc_node_pred_p(self):
        # returns the most probable value *probability* of the label data of some node.

        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        u, c = np.unique(self.data[:, -1], return_counts=True) # np.unique counts unique values
        pred = 100*c.max()/np.sum(c) # takes the max value and divides by sum of all

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        node.parent = self
        node.depth = self.depth+1
        self.children_values.append(val)

    def split(self, impurity_func):
        """
        Splits the current node according to the impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to chi and max_depth.

        Input:
        - The impurity function that should be used as the splitting criteria

        This function has no return value
        """
        ###########################################################################
        # TODO: Implement the function.                                           #
        ###########################################################################
        if impurity_func(self.data) == 0:
            self.prune()
            return
        temp_groups = {}
        temp_goodness = 0.0
        groups = {}
        goodness = -1.0
        attribute = 0
        size_att = self.data.shape[1]-1
        for curr_attribute in range(size_att): # loop over all features
            temp_goodness, temp_groups = goodness_of_split( # check for current feature goodness
                self.data, curr_attribute, impurity_func, self.gain_ratio) 
            if temp_goodness > goodness: # if greater than previous max, replace by current feature
                goodness = temp_goodness
                groups = temp_groups
                attribute = curr_attribute
        self.feature = attribute
        for value in groups: # creates child for each value of feature of split
            self.add_child(DecisionNode(
                groups[value], gain_ratio=self.gain_ratio), value)

        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    
    # helper function, runs over all nodes and returns the maximum
    def max_depth_method(self):
        if not self.children:
            return 0
        else:
            return 1 + max(child.max_depth_method() for child in self.children)

    # helper function to prune some node (turn into leaf)
    def prune(self):
        self.terminal = True
        self.children = []
        self.children_values = []

def build_tree(data, impurity, gain_ratio=False, chi=1, max_depth=1000):
    """
    Build a tree using the given impurity measure and training dataset. 
    You are required to fully grow the tree until all leaves are pure unless
    you are using pruning

    Input:
    - data: the training dataset.
    - impurity: the chosen impurity measure. Notice that you can send a function
                as an argument in python.
    - gain_ratio: goodness of split or gain ratio flag

    Output: the root node of the tree.
    """
    root = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    root = DecisionNode(data, chi=chi, max_depth=max_depth,
                        gain_ratio=gain_ratio)
    q = [root] # init the queue
    while len(q) > 0:
        curr_node = q.pop(0) # pops first in queue
        if curr_node.depth >= max_depth: # checks for max_depth
            curr_node.parent.prune()
            # del curr_node
            continue
        curr_node.split(impurity_func=impurity) # splits the node
        if curr_node != root:
            if curr_node.feature == curr_node.parent.feature: # if redundant split, prune
                curr_node.parent.prune()
                continue
        deg_of_freedom = len(curr_node.children)-1
        if deg_of_freedom > 0 and chi < 1: # checks for chi
            if (calc_chi(curr_node) < chi_table[deg_of_freedom][chi]): # check if the chi after split is high enough, else prune
                curr_node.prune()
                continue
        q += curr_node.children # adds children to queue

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return root


def predict(root, instance):
    """
    Predict a given instance using the decision tree

    Input:
    - root: the root of the decision tree.
    - instance: an row vector from the dataset. Note that the last element 
                of this vector is the label of the instance.

    Output: the prediction of the instance.
    """
    pred = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pred = root.pred
    if len(root.children) > 0: # checks if leaf
        value = instance[root.feature]
        if value in root.children_values:
            index = root.children_values.index(value) # checks for value of instance according to current node's main feature
            root = root.children[index]
            pred = predict(root, instance) # delves deeper to next node
        ###########################################################################
        #                             END OF YOUR CODE                            #
        ###########################################################################
    return pred


def calc_accuracy(node, dataset):
    """
    Predict a given dataset using the decision tree and calculate the accuracy

    Input:
    - node: a node in the decision tree.
    - dataset: the dataset on which the accuracy is evaluated

    Output: the accuracy of the decision tree on the given dataset (%).
    """
    accuracy = 0.0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    accurate_pred = 0
    for inst in dataset: # loop over every instance, if accuracy is good, adds +1 to score
        if predict(node, inst) == inst[-1]:
            accurate_pred += 1
    accuracy = 100*accurate_pred/len(dataset) # returns score / total

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return accuracy


def depth_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously.

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output: the training and testing accuracies per max depth
    """
    training = []
    testing = []
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]: # loops over each depth, builds a new tree and adds the scores to the lists
        tree = build_tree(X_train, calc_entropy,
                          max_depth=max_depth, gain_ratio=True)
        training.append(calc_accuracy(tree, X_train))
        testing.append(calc_accuracy(tree, X_test))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return training, testing


def calc_chi(node):
    chi_squared = 0.0
    P_Y = np.unique(node.data[:, -1], return_counts=True)[1]
    P_Y = P_Y / len(node.data) # calculates the probability for each val
    for val in node.children_values: # loops over all values
        f_data = node.data[node.data[:, node.feature] == val] # all instances for which Y=val & xj=f
        D_f = len(f_data) # number of instances where xj=f
        p_f = np.unique(f_data[:, -1], return_counts=True)[1] # count of f_data
        E = D_f*P_Y
        chi_squared += np.sum(np.divide(np.square(p_f-E), E)) # adds all relevant values calculations vectorized
    return chi_squared


def chi_pruning(X_train, X_test):
    """
    Calculate the training and testing accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_test: the testing data where the last column holds the labels

    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_testing_acc: the testing accuracy per chi value
    - depths: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_testing_acc = []
    depth = []
    tree = []
    index = 0
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    for chi_val in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]: # loops over all chi values, builds corresponding tree, calculates accuracy and appends to lists
        tree.append(build_tree(X_train, calc_entropy,
                    chi=chi_val, gain_ratio=True))
        chi_training_acc.append(calc_accuracy(tree[index], X_train))
        chi_testing_acc.append(calc_accuracy(tree[index], X_test))
        depth.append(tree[index].max_depth_method())
        index += 1
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node): # loops over all nodes to count
    """
    Count the number of node in a given tree

    Input:
    - node: a node in the decision tree.

    Output: the number of nodes in the tree.
    """
    n_nodes = 1
    q = [node]
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    while (len(q)):
        q += q[0].children
        n_nodes += len(q[0].children)
        q.pop(0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes


def print_tree(node, depth=0, parent_feature='ROOT', feature_val='ROOT'):
    '''
    prints the tree according to the example above
    Input:
    - node: a node in the decision tree
    This function has no return value
    '''
    if node.terminal == False:
        if node.depth == 0:
            print('[ROOT, feature=X{}, num child: {}, data len: {}, pred: {}]'.format(
                node.feature, len(node.children), len(node.data), node.calc_node_pred()))
        else:
            print('{}[X{}={}, feature=X{}], Depth: {}, num child: {}, data len: {}'.format(depth*'--', parent_feature, feature_val,
                                                                                           node.feature, node.depth, len(node.children), len(node.data)))
        for i, child in enumerate(node.children):
            print_tree(child, depth+1, node.feature, node.children_values[i])
    else:
        classes_count = {}
        labels, counts = np.unique(node.data[:, -1], return_counts=True)
        for l, c in zip(labels, counts):
            classes_count[l] = c
        print('{}[X{}={}, leaf]: [{}], Depth: {}, data len: {}, pred: {}, {:.1f}%]'.format(depth*'--', parent_feature, feature_val,
                                                                                  classes_count, node.depth, len(node.data), node.calc_node_pred(), node.calc_node_pred_p()))


def print_depth(node, depth):
    q = [node]
    while (len(q) > 0):
        if q[0].depth < depth:
            q += q[0].children
        elif q[0].depth == depth:
            if depth != 0:
                print('{}[X{}={}, feature=X{}], Depth: {}, num child: {}, data len: {}, pred: {}, {:.2f}%'.format(
                    depth*'--', q[0].parent.feature, q[0].feature, q[0].feature, q[0].depth, len(q[0].children), len(q[0].data), q[0].calc_node_pred(), q[0].calc_node_pred_p()))
            else:
                print('[ROOT, feature=X{}, num child: {}, data len: {}, pred: {}]'.format(
                    q[0].feature, len(q[0].children), len(q[0].data), q[0].calc_node_pred()))
        q.pop(0)
