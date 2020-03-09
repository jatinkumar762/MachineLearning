import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *
from mpl_toolkits import mplot3d

np.random.seed(42)

N = 30
P = 1
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

LR = LinearRegression(fit_intercept=True)
LR.fit_vectorised(X, y,n_iter=10,batch_size=10)

Ltheta= np.asarray(LR.theta_list)

LR.err.clear()
i=1
for lth in Ltheta:
    LR.plot_surface(X,y,lth[0],lth[1],lth,i)
    i+=1

i=1
for lth in Ltheta:
    LR.plot_line_fit(X,y,lth[0],lth[1],lth,i)
    i+=1

LR.err.clear()
x_bias =np.ones((X.shape[0],1))
X = np.append(x_bias,X,axis=1)

i=0
for lth in Ltheta:
    if i%5==0:
        error = LR.error(X, y, lth)
        LR.err.append(error)
    i+=1

i=1
for lth in Ltheta:
    LR.plot_contour(X,y,lth[0],lth[1],LR.err,i)
    i+=1
    break

