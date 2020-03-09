import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression

from metrics import *
import timeit


np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

for fit_intercept in [True, False]:
    print('fit_intercept: ',fit_intercept)
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = pd.Series(LR.predict(X))
    LR.y_hat = y_hat
    LR.plot_residuals()
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print()


No_Of_Samples = [100,200,500]
time = []
for nos in No_Of_Samples:
    N = nos
    P = 5
    X = pd.DataFrame(np.random.randn(N, P))
    y = pd.Series(np.random.randn(N))

    for fit_intercept in [True]:
        LR = LinearRegression(fit_intercept=fit_intercept)
        start = timeit.default_timer()
        LR.fit(X, y)
        y_hat = pd.Series(LR.predict(X))
        stop = timeit.default_timer()
        print('Time to Solve using linear regression ',N ,'samples: ', stop - start)
        time.append(stop - start)


line, =plt.plot(No_Of_Samples,time, label='Time to Solve')
plt.title('Comparison for different input size data')
plt.xlabel('Samples in Data')
plt.ylabel('Time in sec.')
plt.legend()
plt.show()





# Assume N  training examples and C features
# O(C*C*N)  to multiply XT by X
# O(C*N) to multiply XT by Y
# O(C*C*C) to compute (XTX)−1(XTY)
# assume that N>C 
# Therefore the total time complexity is  O(C*C*N)