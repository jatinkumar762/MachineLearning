"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index
from sklearn import tree

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth=1000):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"} # criterion won't be used for regression
        > max_depth : The maximum depth the tree can grow to 
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.output = ""
        self.input = ""
        self.tree = ""
        

    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        if self.output=="discrete":
           crt = 'entropy' if self.criterion == 'information_gain' else 'gini'
           clf = tree.DecisionTreeClassifier(criterion=crt,max_depth=self.max_depth)
           clf = clf.fit(X,y)
           self.tree = clf
        elif self.output=="real":
           crt = 'mse' if self.criterion == 'mse' else 'mae'
           clf = tree.DecisionTreeRegressor(criterion=crt,max_depth=self.max_depth)
           clf.fit(X,y)
           self.tree = clf
           

    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        return self.tree.predict(X)

    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
    