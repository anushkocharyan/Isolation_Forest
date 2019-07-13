import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def c(n):
    if n > 2:
        return 2*(np.log(n-1)+0.5772156649) - (2*(n-1)/n)
    elif n == 2:
        return 1
    return 0


class IsolationTreeEnsemble:
    def __init__(self, sample_size, n_trees=10):
        self.sample_size = sample_size
        self.n_trees = n_trees
        self.trees = []
        self.c = c(sample_size)
        self.hl = np.ceil(np.log2(sample_size))

    def fit(self, X:np.ndarray):
        """
        Given a 2D matrix of observations, create an ensemble of IsolationTree
        objects and store them in a list: self.trees.  Convert DataFrames to
        ndarray objects.
        """

        if isinstance(X, pd.DataFrame):
            X = X.values

        for i in range(self.n_trees):
            iTree = IsolationTree(self.hl)
            iTree.fit(X[np.random.choice(X.shape[0], size=self.sample_size, replace=False), :])
            self.trees.append(iTree)


    def path_length(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the average path length
        for each observation in X.  Compute the path length for x_i using every
        tree in self.trees then compute the average for each x_i.  Return an
        ndarray of shape (len(X),1).
        """

        size = len(X)
        avg_list = np.empty(size)

        second  = 2*(np.log(size-1)+0.5772156649) - (2*(size-1)/size)
        if isinstance(X, pd.DataFrame):
            X = X.values
        i = 0
        for x in X:
            xAvgLen = 0
            for t in self.trees:
                xAvgLen += t.getNode(x)
            Eh = xAvgLen/self.n_trees + second
            avg_list[i] = 2.0**(-Eh/self.c)
            i += 1

        return avg_list

    def anomaly_score(self, X:np.ndarray) -> np.ndarray:
        """
        Given a 2D matrix of observations, X, compute the anomaly score
        for each x_i observation, returning an ndarray of them.
        """
        return self.path_length(X)


    def predict_from_anomaly_scores(self, scores:np.ndarray, threshold:float) -> np.ndarray:
        """
        Given an array of scores and a score threshold, return an array of
        the predictions: 1 for any score >= the threshold and 0 otherwise.
        """
        predict = np.zeros(len(scores))
        for i in range(len(scores)):
            if scores[i] >= threshold:
                predict[i] = 1
            else: predict[i] = 0

        return predict

    def predict(self, X:np.ndarray, threshold:float) -> np.ndarray:
        "A shorthand for calling anomaly_score() and predict_from_anomaly_scores()."
        return self.predict_from_anomaly_scores(self.anomaly_score(X), threshold)


class TreeNode:
    def __init__(self, path_len=0, split_point=None, feature_index=None, left=None, right=None, isExt=False):
        self.path_len = path_len    
        self.split_point = split_point
        self.feature_index = feature_index
        self.left = left
        self.right = right
        self.isExt = isExt

    def get_node(self, node):
        if self.isExt:
            return self.path_len

        indx = self.feature_index
        if node[indx] < self.split_point:
            return self.left.get_node(node)
        return self.right.get_node(node)


class IsolationTree:
    def __init__(self, height_limit):
        self.height_limit = height_limit
        self.root = None
        self.n_nodes = 0

    def fit(self, X, height=0):
        if height >= self.height_limit or len(X) <= 1:
            self.root = TreeNode(path_len=height, isExt=True)
            self.n_nodes +=1
            return self.root
        else:
            feature_index = np.random.randint(0, X.shape[1])
            minimum = int(X[:,feature_index].min())
            maximum = int(X[:,feature_index].max())

            if minimum == maximum:
                self.root = TreeNode(path_len=height, split_point=minimum, feature_index=feature_index, isExt=True)
                self.n_nodes +=1
                return self.root
            split_point = np.random.uniform(minimum, maximum)

            left = X[X[:,feature_index] < split_point]

            right = X[X[:,feature_index] >= split_point]
            
            self.n_nodes +=1
            self.root = TreeNode(path_len=height, split_point=split_point,
                feature_index=feature_index,
                left=self.fit(left, height + 1),
                right=self.fit(right, height + 1), isExt=False)
            return self.root


    def getNode(self, node):
        if self.root == None: return None
        return self.root.get_node(node)


def find_TPR_threshold(y, scores, desired_TPR):
    """
    Start at score threshold 1.0 and work down until we hit desired TPR.
    Step by 0.01 score increments. For each threshold, compute the TPR
    and FPR to see if we've reached to the desired TPR. If so, return the
    score threshold and FPR.
    """
    threshold = 1

    while threshold !=0:
        predictions = []
        for s in scores:
            if s >= threshold: predictions.append(1)
            else: predictions.append(0)
        confusion = confusion_matrix(y, predictions)
        TN, FP, FN, TP = confusion.flat
        TPR = TP / (TP + FN)
        FPR = FP / (FP + TN)
        if TPR >= desired_TPR:
            return threshold, FPR
        threshold -= 0.01
