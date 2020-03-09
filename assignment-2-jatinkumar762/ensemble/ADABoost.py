import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tree.myBase import DecisionTree
from sklearn import tree as sktree


class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimator = n_estimators
        self.modals = []
        self.alphas = []
        self.column = []
        self.weight = ""
        self.X = None
        self.y = None
        self.res = None

    def final_visualize_classifier(self, X, y, ax=None, cmap='rainbow'):
        ax = ax or plt.gca()
        
        y[y>=0] = 1
        y[y<0] = 0

        # Plot the training points
        scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # fit the estimator
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
 
 
        Z = self.res
        
        Z[Z>=0] = 1
        Z[Z<0] = 0
        
        # # Create a color plot with the results
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)
        
        legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title="Classes")
        ax.add_artist(legend1)
        ax.set_title('Final Estimator')
        ax.set_xlabel("Feature - X1")
        ax.set_ylabel("Feature - X2")
        ax.set(xlim=xlim, ylim=ylim)


    def visualize_classifier(self,model, X, y, index, alpha, ax=None, cmap='rainbow'):
        ax = ax or plt.gca()
        
        #change -1 to 0
        y[y>=0] = 1
        y[y<0] = 0

        # Plot the training points
        scatter = ax.scatter(X[:, model.splitColumn], X[:, (1-model.splitColumn)], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
        ax.axis('tight')

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # fit the estimator
        xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
 
 
        Z = alpha*model.predict(pd.DataFrame(xx.ravel())).astype(int).to_numpy().reshape(xx.shape)

        if index == 1:
            self.res = Z.copy()
        else:
            self.res += Z

        Z[Z>=0] = 1
        Z[Z<0] = 0
        
        # # Create a color plot with the results
        n_classes = len(np.unique(y))
        contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)
        
        legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title="Classes")
        ax.add_artist(legend1)
        ax.set_title('Estimator No: '+str(index))
        ax.set_xlabel("Feature - X1")
        ax.set_ylabel("Feature - X2")
        ax.set(xlim=xlim, ylim=ylim)

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        #base = self.base_estimator
        N,_ = X.shape
        W = np.ones(N)/N
        y = y.replace(0,-1)

        self.modals = []
        self.alphas = []
        self.weight = ""
        self.X = X
        self.y = y

        for i in range(self.n_estimator):
            base = DecisionTree(criterion='information_gain',max_depth=1)
            base.output = 'category'
            base.input = 'real'
            base.fit(X.copy(),y.copy(),W.copy())
            #print(base.plot())
            #print('Column ',base.splitColumn)
            #print(X[base.splitColumn])
            #print(base.predict(pd.DataFrame(X[base.splitColumn])))
            self.column.append(base.splitColumn)
            p = base.predict(pd.DataFrame(X[base.splitColumn].copy())).astype(int)
            #print(p)
            w_sum  = np.sum(W)
            err=0
            for i in range(N):
                if (p.iloc[i]!=y.iloc[i]).any():
                    err+=W[i]

            err = err/w_sum
            alpha = 0.5*(np.log(1-err) - np.log(err))

            for i in range(N):
                if (p.iloc[i] == y.iloc[i]).any():
                    W[i] = W[i] * np.exp(-1*alpha)
                else:
                    W[i] = W[i] * np.exp(alpha)
            
            W = W/np.sum(W)

            self.modals.append(base)
            self.alphas.append(alpha)

        self.weight = W

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        N,_ = X.shape
        FX= np.zeros(N)
        count = 0
        for alpha,tree in zip(self.alphas,self.modals):
            FX+=alpha*tree.predict(pd.DataFrame(X[tree.splitColumn])).astype(int)
        
        FX[FX>=0] = 1
        FX[FX<0] = 0
        
        return pd.Series(FX.astype(int))
  

    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """

        i=1
        for tree,alpha in zip(self.modals,self.alphas):
            plt.subplot(1,3,i)
            self.visualize_classifier(tree, self.X.to_numpy(), self.y.to_numpy(),i,alpha)
            i+=1
        plt.show()

        self.final_visualize_classifier(self.X.to_numpy(), self.y.to_numpy())
        plt.show()
