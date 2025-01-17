"""
The current code given is for the Assignment 2.
> Classification
> Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 3
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'information_gain'
tree = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
Classifier_AB.plot()
# #[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
print()
for cls in y.unique():
    print('Category: ',cls)
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
    print()
