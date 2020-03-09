import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression
from metrics import *
import pandas as pd


# x = np.array([i*np.pi/180 for i in range(60,300,4)])
# np.random.seed(10)  #Setting seed for reproducibility
# y = 4*x + 7 + np.random.normal(0,3,len(x))

degree = [1, 3, 5, 7, 9]
fnl_theta = []

N = [300,400,500]

for n in N:
    temp=[]
    temp.clear()
    for d in degree:
        x = np.array([i*np.pi/180 for i in range(60,n,4)])
        poly = PolynomialFeatures(d)
        poly.transform(x)
        newx = np.asarray(poly.result)
        new_X = newx[:, np.newaxis]
        new_Y = 4*newx + 7 + np.random.normal(0,3,len(newx))
        for fit_intercept in [False]:
            LR = LinearRegression(fit_intercept=fit_intercept)
            LR.fit_vectorised(pd.DataFrame(new_X), pd.Series(new_Y), n_iter=5) # here you can use fit_non_vectorised / fit_autograd methods
            temp.append(LR.coef_[0])
    fnl_theta.append(temp)

print(fnl_theta)

fig, ax = plt.subplots()
i=0
for n in N:
    ax.plot(degree, fnl_theta[i],label='N :'+str(len(list(range(60,n,4)))))
    ax.set_yscale('log')
    i=i+1

plt.xlabel('degree')
plt.ylabel('magnitude of theta')
legend = ax.legend(loc='upper center', shadow=True)
plt.show()