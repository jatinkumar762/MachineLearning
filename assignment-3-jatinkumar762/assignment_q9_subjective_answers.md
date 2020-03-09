# ES654-2020 Assignment 3

*Jatin Kumar* - *19210045*

------

> Write the answers for the subjective questions here

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

X[1] = 2*X[0] 
X[3] = X[0] + X[2]

fit_vectorised and fit_intercept: False
RMSE:  0.9116751593601505
MAE:  0.6529821831151797

gradient descent implementation working for the data set that suffers from multicollinearity