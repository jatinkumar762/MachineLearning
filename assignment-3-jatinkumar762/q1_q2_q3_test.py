
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_vectorised(X, y,batch_size=10) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('fit_vectorised and fit_intercept:',fit_intercept)
    print('RMSE: ', rmse(pd.Series(y_hat), y))
    print('MAE: ', mae(pd.Series(y_hat), y))
    print()


for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_non_vectorised(X, y,batch_size=10) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('fit_non_vectorised and fit_intercept:',fit_intercept)
    print('RMSE: ', rmse(pd.Series(y_hat), y))
    print('MAE: ', mae(pd.Series(y_hat), y))
    print()

for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit_autograd(X, y) # here you can use fit_non_vectorised / fit_autograd methods
    y_hat = LR.predict(X)
    print('fit_non_vectorised and fit_intercept:',fit_intercept)
    print('RMSE: ', rmse(pd.Series(y_hat), y))
    print('MAE: ', mae(pd.Series(y_hat), y))
    print()