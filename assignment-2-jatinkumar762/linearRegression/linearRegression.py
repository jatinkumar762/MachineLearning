import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

class LinearRegression():
    def __init__(self, fit_intercept=True, method='normal'):
        '''

        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        :param method:
        '''
        self.fit_intercept = fit_intercept
        self.theta = None
        self.X = None
        self.y = None
        self.y_hat = None

    def fit(self, X, y):
        '''
        Function to train and construct the LinearRegression
        :param X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        :param y: pd.Series with rows corresponding to output variable (shape of Y is N)
        :return:
        '''
        N,col = X.shape

        if self.fit_intercept == True:
            x_bias = np.ones((N,1))
            X = np.append(x_bias,X,axis=1)

        x_transpose = np.transpose(X)
        x_trans_dot_x = x_transpose.dot(X)
        temp1 = np.linalg.pinv(x_trans_dot_x)
        temp2 = x_transpose.dot(y)
        self.theta = temp1.dot(temp2)
        self.X = X
        self.y = y

    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point
        :param X: pd.DataFrame with rows as samples and columns as features
        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''
        N,col =X.shape
        if self.fit_intercept == True:
            x_bias = np.ones((N,1)).astype('int')
            X = np.append(x_bias,X,axis=1)

        return X.dot(self.theta)
        

    def plot_residuals(self):
        """
        Function to plot the residuals for LinearRegression on the train set and the fit. This method can only be called when `fit` has been earlier invoked.

        This should plot a figure with 1 row and 3 columns
        Column 1 is a scatter plot of ground truth(y) and estimate(\hat{y})
        Column 2 is a histogram/KDE plot of the residuals and the title is the mean and the variance
        Column 3 plots a bar plot on a log scale showing the coefficients of the different features and the intercept term (\theta_i)

        """
        clr = np.ones(len(self.y))
        plt.scatter(self.y, self.y_hat, s=80, c=clr, marker='+')
        plt.title('Fit Intercept: '+str(self.fit_intercept))
        plt.show()

        residuals = (self.y - pd.Series(self.y_hat)).abs().to_numpy()
        plt.title('Fit Intercept: '+str(self.fit_intercept))
        plt.hist(residuals)
        plt.show()

        #print([*range(0,len(pd.DataFrame(self.X).columns))])
        if self.fit_intercept == True:
             xAxis = list(range(0,len(pd.DataFrame(self.X).columns)))
             print(xAxis)
             print(self.theta)
             plt.bar(xAxis, self.theta)
             plt.yscale("log")
        else:
            xAxis = list(range(1,len(pd.DataFrame(self.X).columns)+1))
            plt.bar(xAxis, self.theta)
            plt.yscale("log")
        plt.show()


