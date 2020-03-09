import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

N = 30
P = 5
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size = N), dtype="category")

print('\n\n##Discrete Input and Discrete Output##')
for criteria in ['information_gain']:
    tree = DecisionTree(criterion=criteria,max_depth=np.inf) #Split based on Inf. Gain
    tree.fit(X, y)
    y_hat = tree.predict(X)
    tree.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print(cls)
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))


