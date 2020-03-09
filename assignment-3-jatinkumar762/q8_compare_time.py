import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
import timeit

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

time_gd = None
time_nrm = None

for fit_intercept in [True]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    start = timeit.default_timer()
    LR.fit_vectorised(X.copy(), y.copy()) # here you can use fit_non_vectorised / fit_autograd methods
    stop = timeit.default_timer()
    y_hat = LR.predict(X)
    time_gd = stop - start
    start = timeit.default_timer()
    LR.fit_normal(X.copy(),y.copy())
    stop = timeit.default_timer()
    time_nrm = stop - start
    y_hat = LR.predict(X)

print('Using gradient:',time_gd)
print('Using Normal Equation',time_nrm)





