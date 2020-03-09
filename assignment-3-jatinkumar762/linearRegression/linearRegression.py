import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Import Autograd modules here
from autograd import grad
import autograd.numpy as autonp 

class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods
        self.theta_list = []
        self.err = []

    def fit_non_vectorised(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        rows, cols = X.shape
        theta = None
        der_list = []
        LR = lr

        if self.fit_intercept == True:
            theta = np.ones(cols+1)
            x_bias =np.ones((rows,1))
            X = np.append(x_bias,X,axis=1)
            cols = cols+1
        else:
            theta = np.ones(cols)
            X = X.to_numpy()

        if batch_size == 1:
            i_c=1
            for itr in range(n_iter):
                r_i = np.random.randint(0,rows)
                X_i = X[r_i,:]
                y_i = y[r_i]

                dev_list = []
                dev_list.clear()
                for k in range(len(X_i)):
                    der_sum=0
                    hype_i=0
                    for j in range(len(X_i)):
                        hype_i+=theta[j]*X_i[j]
                    der_sum = (hype_i-y_i)*X_i[k]
                    dev_list.append(der_sum)
                
                if lr_type=='inverse':
                    lr = LR/i_c
                theta = theta - lr*np.array(dev_list)
                i_c+=1
        else:
            indices = np.random.permutation(rows)
            X = X[indices]
            y = y[indices]

            i_c=1
            for itr in range(n_iter):
                for i in range(0,rows,batch_size):
                    X_i = X[i:i+batch_size]
                    y_i = y[i:i+batch_size]
                    dev_list = []
                    dev_list.clear()

                    for k in range(X_i.shape[1]):
                        der_sum=0
                        for i in range(X_i.shape[0]):
                            hype_i=0
                            for j in range(X_i.shape[1]):
                                hype_i+=theta[j]*X_i[i,j]
                            der_i = (hype_i - y[i])*X_i[i,k]
                            der_sum += der_i
                        der_sum = (1/X_i.shape[0])*der_sum
                        dev_list.append(der_sum)

                    if lr_type=='inverse':
                         lr = LR/i_c
                    theta = theta - lr*np.array(dev_list)
                i_c+=1


            # for itr in range(n_iter):
            #     dev_list = []
            #     dev_list.clear()
            #     for k in range(cols):
            #         der_sum=0
            #         for i in range(rows):
            #             hype_i=0
            #             for j in range(cols):
            #                 hype_i+=theta[j]*X[i,j]
            #             der_i = (hype_i - y[i])*X[i,k]
            #             der_sum += der_i
            #         der_sum = (1/rows)*der_sum
            #         dev_list.append(der_sum)
            #     theta = theta - lr*np.array(dev_list)
       
        self.coef_=theta

    def fit_vectorised(self, X, y,batch_size=1, n_iter=1000, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        rows, cols = X.shape
        theta = None
        LR = lr
        self.theta_list.clear()

        if self.fit_intercept == True:
            theta = np.ones(cols+1)
            x_bias =np.ones((rows,1))
            X = np.append(x_bias,X,axis=1)
            cols = cols+1
        else:
            theta = np.ones(cols)
            X = X.to_numpy()

        if batch_size==1:
            i_c = 1
            for itr in range(n_iter):
                r_i = np.random.randint(0,rows)
                X_i = X[r_i,:]
                y_i = y[r_i]

                if lr_type=='inverse':
                    lr = LR/i_c
                theta = theta - lr*(1/rows)*(np.transpose(X_i).dot(X_i.dot(theta) - y_i))
                self.theta_list.append(theta)
                i_c+=1
        else:
            indices = np.random.permutation(rows)
            X = X[indices]
            y = y[indices]
            i_c = 1
            for itr in range(n_iter):
                for i in range(0,rows,batch_size):
                    X_i = X[i:i+batch_size]
                    y_i = y[i:i+batch_size]
                    if lr_type=='inverse':
                        lr = LR/i_c
                    theta = theta - lr*(1/rows)*(np.transpose(X).dot(X.dot(theta) - y))
                    self.theta_list.append(theta)
                i_c+=1

        self.coef_=theta

    def mse(self,theta,X,y):  
        N,_= X.shape
        xtheta = autonp.dot(X,theta)
        se = np.sum(autonp.power(y - xtheta,2))/N
        return se


    def fit_autograd(self, X, y, batch_size=1, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''
        rows, cols = X.shape
        theta = None

        if self.fit_intercept == True:
            theta = np.ones(cols+1)
            x_bias =np.ones((rows,1))
            X = np.append(x_bias,X,axis=1)
            cols = cols+1
        else:
            theta = np.ones(cols)
            X = X.to_numpy()
        
        g_me = grad(self.mse)
        for itr in range(n_iter):        
            theta = theta - lr*(1/rows)*g_me(theta,X,y)
        self.coef_=theta

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
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
        rows, cols = X.shape
        if self.fit_intercept == True:
            x_bias =np.ones((rows,1))
            X = np.append(x_bias,X,axis=1)
        return X.dot(self.coef_)

    def error(self,X, Y, THETA):
        return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)

    def plot_surface(self, X, y, t_0, t_1, theta,index):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        ms = np.linspace(t_0 - 1 , t_0 + 1, 10)
        bs = np.linspace(t_1 - 1 , t_1 + 1, 20)

        M, B = np.meshgrid(ms, bs)
  
        x_bias =np.ones((X.shape[0],1))
        X = np.append(x_bias,X,axis=1)

        error = self.error(X, y, theta)
        self.err.append(error)


        zs = np.array([self.error(X, y, theta) for theta in zip(np.ravel(M), np.ravel(B))])
        Z = zs.reshape(M.shape)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.2)
        ax.set_xlabel('M', labelpad=30, fontsize=14, fontweight='bold')
        ax.set_ylabel('B', labelpad=30, fontsize=14, fontweight='bold')
        ax.set_zlabel('Error', labelpad=30, fontsize=14, fontweight='bold')
        ax.view_init(elev=20., azim=30)
        fig.suptitle(str('Error:{:.5f}'.format(error)), fontsize=14, fontweight='bold')
        ax.plot([t_0], [t_1], error , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7)
        #ax.plot([history[0][0]], [history[0][1]], [cost[0]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7)
        plt.savefig("surface"+str(index)+".png")


    def plot_line_fit(self, X, y, t_0, t_1, theta,index):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        x_bias =np.ones((X.shape[0],1))
        NX = np.append(x_bias,X,axis=1)

        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        ax.scatter(X, y, marker='o', s=40, color='b')
        ax.set_xlabel('X', fontsize=14, fontweight='bold')
        ax.set_ylabel('Y', fontsize=14, fontweight='bold')
        ax.plot(X, np.dot(NX,theta), color='r', lw=1)
        fig.suptitle(str('theta_0:{:.5f} theta_1:{:.5f}'.format(t_0,t_1)), fontsize=14, fontweight='bold')
        plt.savefig("linefit"+str(index)+".png")

    def plot_contour(self, X, y, t_0, t_1, errors,index):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        theta0 = np.linspace(-t_0 *1.5 , t_0 *1.5, 100)
        theta1 = np.linspace(-t_1 *2 , t_1 *2, 100)

        T0, T1 = np.meshgrid(theta0, theta1)

        #print(errors)
        levels = np.sort(np.array(errors))

        zs = np.array([self.error(X, y, theta) for theta in zip(np.ravel(T0), np.ravel(T1))])
        Z = zs.reshape(T0.shape)

        fig,ax=plt.subplots(1,1)
        cp = ax.contour(T0, T1, Z,levels,alpha=.7)
        plt.axhline(0, color='black', alpha=.5, dashes=[2, 4],linewidth=1)
        plt.axvline(0, color='black', alpha=0.5, dashes=[2, 4],linewidth=1)
        plt.clabel(cp, inline=1, fontsize=10)
        plt.xlabel('theta0')
        plt.ylabel('theta1')

        Ltheta= np.asarray(self.theta_list)
        i=0
        p0, p1 = None, None
        for lth in Ltheta:
            if i%5==0:
                if i==0:
                    p0,p1 = lth[0],lth[1]
                else:
                    plt.annotate('', xy=(lth[0],lth[1]), xytext=(p0,p1), arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 0.5},va='center', ha='center')
                    p0,p1 = lth[0],lth[1]
            i+=1
        plt.show()