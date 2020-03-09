import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

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
train_data,test_data = train_test_split(df)
X = train_data[['sepal_width','petal_width']]
y = train_data[['label']]['label']

for criteria in ['information_gain', 'gini_index']:
    Classifier_RF = RandomForestClassifier(50, criterion = criteria)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    #Classifier_RF.plot()
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    print()
    for cls in y.unique():
            print('Class: ',cls)
            print('Precision: ', precision(y_hat, y, cls))
            print('Recall: ', recall(y_hat, y, cls))
            print()
