from .base import DecisionTree
import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.tree.export import export_text
from sklearn.tree import DecisionTreeClassifier

def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
        ax = ax or plt.gca()
        
        # Plot the training points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')
        #ax.axis('off')
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        
        #model.fit(X, y)
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

        # Create a color plot with the results
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)
        
        legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title="Classes")
        ax.add_artist(legend1)
        ax.set(xlim=xlim, ylim=ylim)

class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=100):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''      
        self.n_estimators = n_estimators
        self.criteron = criterion
        self.max_depth = max_depth
        self.models = []
        self.col_names = []
        self.X = None
        self.y = None

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        N,col = X.shape
        
        for i in range(self.n_estimators):
            base = DecisionTree(criterion=self.criteron)
            base.output = 'discrete'
            base.input = 'real'
            tmp = X.copy()
            if(col>2):
                tmp = tmp.sample(n=col-3,replace= False,axis=1)
            else:
                tmp = tmp.sample(n=col,replace= True,axis=1)

            #print(list(tmp.columns))
            base.fit(tmp,y)
            self.models.append(base)
            self.col_names.append(list(tmp.columns))
            self.X = X
            self.y = y

        
    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        output = {}
        result = ""
        tmp = X.copy()
        _,col=X.shape
        i=0
        for tree, col in zip(self.models,self.col_names):
            col_n = "tree_{}".format(i)
            output[col_n] = tree.predict(X[col])
            
        output = pd.DataFrame(output)
        result = output.mode(axis=1)[0]
        return result

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        #X = self.X.values
        #y = self.y.to_numpy()

        # tree.plot_tree(self.models[0].tree)
        # plt.show()
        figure = plt.figure(figsize=(13, 5))
        figure.subplots_adjust(hspace=0.4, wspace=0.4)
        i=1
        for t in self.models:
            ax = plt.subplot(1,2,i)
            #r = export_text(t.tree)
            #print(r)
            #ax.text(0.5, 0.5, r, fontsize=18, ha='center')
            tree.plot_tree(t.tree)
            ax.set_title('Estimator: '+str(i))
            i=i+1
            if i==3:
                break
        plt.show()

        #y = self.y.replace({'label' : {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1}})
        i=1
        for t, col in zip(self.models,self.col_names):
             ax = plt.subplot(1,2,i)
             visualize_classifier(t, self.X[col].values, self.y.to_numpy())
             ax.set_title('Estimator: '+str(i))
             i=i+1
             if i==3:
                 break
        plt.show()
        #r = export_text(self.models[0].tree)
        #print(r)


class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='variance', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''

        self.n_estimators = n_estimators
        self.criteron = criterion
        self.max_depth = max_depth
        self.models = []
        self.col_names = []

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        N,col = X.shape
       
        for i in range(self.n_estimators):
            base = DecisionTree(criterion=self.criteron)
            base.output = 'real'
            base.input = 'real'
            tmp = X.copy()
            tmp = tmp.sample(n=col-3,replace= False,axis=1)
            #print(list(tmp.columns))
            base.fit(tmp,y)
            self.models.append(base)
            self.col_names.append(list(tmp.columns))

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        output = {}
        result = ""
        tmp = X.copy()
        _,col=X.shape
        i=0
        for tree, col in zip(self.models,self.col_names):
            col_n = "tree_{}".format(i)
            output[col_n] = tree.predict(X[col])
            
        output = pd.DataFrame(output)
        result = output.mode(axis=1)[0]
        return result

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        pass
