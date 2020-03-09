import numpy as np
import pandas as pd
from tree.base import DecisionTree
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_classifier(model, X, y, index, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap, clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    # #ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # # fit the estimator
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200), np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3, levels=np.arange(n_classes + 1) - 0.5, cmap=cmap, clim=(y.min(), y.max()), zorder=1)
    
    legend1 = ax.legend(*scatter.legend_elements(),loc="upper right", title="Classes")
    ax.add_artist(legend1)
    ax.set_title('Estimator No: '+str(index))
    ax.set_xlabel("Feature - X1")
    ax.set_ylabel("Feature - X2")
    ax.set(xlim=xlim, ylim=ylim)
    

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.base_estimator = base_estimator
        self.n_estimator = n_estimators
        self.models = []
        self.X = None
        self.y = None
        self.y_hat = []

    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        
        self.X = X
        self.y = y

        XN = X.copy()
        XN['label'] = y
        N,col=X.shape
        for i in range(self.n_estimator):
            base = DecisionTree(criterion='information_gain')
            base.output = 'discrete'
            base.input = 'real'
            tmp = XN.sample(n = N, replace= True) 
            base.fit(tmp.iloc[:,:col],tmp['label'])
            self.models.append(base)


    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        #output = pd.DataFrame(columns=list(range(0,len(self.models))))
        output = ""
        result = ""
        for key, tree in enumerate(self.models):
            if key==0:
               output = pd.Series(tree.predict(X))
               self.y_hat.append(output)
            else:
               temp = pd.Series(tree.predict(X))
               self.y_hat.append(temp)
               output = pd.concat([output, temp],axis=1)

        result = pd.Series(output.max(axis=1))
        return result

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        i=1
        for tree,y_hat in zip(self.models,self.y_hat):
            plt.subplot(130+i)
            visualize_classifier(tree, self.X.to_numpy(), self.y.to_numpy(),i)
            i+=1
        plt.show()
