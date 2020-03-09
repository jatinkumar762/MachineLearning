import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from metrics import *
import pandas as pd


# x = np.array([i*np.pi/180 for i in range(60,300,4)])
# np.random.seed(10)  #Setting seed for reproducibility
# y = 4*x + 7 + np.random.normal(0,3,len(x))

# def f(x):
#     4*x + 7 + np.random.normal(0,3,len(x))

degree = [2,4,6,8,10]
fnl_theta = []

for d in degree:
    x = np.array([i*np.pi/180 for i in range(60,300,4)])
    poly = PolynomialFeatures(d)
    poly.transform(x)
    newx = np.asarray(poly.result)
    new_X = newx[:, np.newaxis]
    new_Y = 4*newx + 7 + np.random.normal(0,3,len(newx))
    for fit_intercept in [False]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        LR.fit_vectorised(pd.DataFrame(new_X), pd.Series(new_Y), n_iter=5) # here you can use fit_non_vectorised / fit_autograd methods
        fnl_theta.append(np.absolute(LR.coef_[0]))
    
print(fnl_theta)

plt.plot(degree, fnl_theta)
plt.yscale('log')
plt.xlabel('degree')
plt.ylabel('magnitude of theta')
plt.show()

