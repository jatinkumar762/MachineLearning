import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree
# Or you could import sklearn DecisionTree
from linearRegression.linearRegression import LinearRegression

np.random.seed(42)

NUM_OP_CLASSES = 2
n_estimators = 3
criteria = 'information_gain'

def train_test_split(df):
    index=df.index.tolist()
    np.random.shuffle(index)
    len,_=df.values.shape
    train_size=int(len*0.6)
    train_data=df.loc[index][0:train_size]
    test_data=df.loc[index][train_size:]  
    return train_data,test_data


df=pd.read_csv("iris.data",names=['sepal_length','sepal_width','petal_length','petal_width','label'])
df = df[['sepal_width','petal_width', 'label']]
df = df.replace({'label' : {'Iris-setosa': 0, 'Iris-versicolor': 0, 'Iris-virginica': 1}})
train_data,test_data = train_test_split(df)
X = train_data[['sepal_width','petal_width']]
y = train_data[['label']]['label']

tree_iris = DecisionTree(criterion=criteria,max_depth=1)
Classifier_AB_iris = AdaBoostClassifier(base_estimator=tree_iris, n_estimators=n_estimators )
Classifier_AB_iris.fit(X, y)
y_hat = Classifier_AB_iris.predict(X)
#[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
print()
for cls in y.unique():
    if cls == 1:
        print('Category: Iris-virginica')
    else:
        print('Category: Not Iris-virginica')
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
    print()