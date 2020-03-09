import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from pprint import pprint

np.random.seed(42)

# Read IRIS data set
# ...
# 

tree = DecisionTree(criterion='information_gain',max_depth=10) #Split based on Inf. Gain
tree.output="category"
tree.input="real"
df=pd.read_csv("iris.data",names=['sepal_length','sepal_width','petal_length','petal_width','label'])
train_data,test_data=tree.train_test_split(df)
sub_tree = tree.decision_tree_algorithm(train_data)
tree.tree=sub_tree
rows,colums=test_data.values.shape
y_hat = tree.predict(test_data.iloc[:,0:colums-1])
y= test_data.iloc[:,-1]
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print('Class Name: ',cls)
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
    print()


index=df.index.tolist()
len,_=df.values.shape
#print(len)
test_size=int(len*0.2)
for i in range(5):   

    if i==0:
        test_finish = len
        test_begin = len-test_size 
    else:
        test_finish = test_begin
        test_begin = test_finish - test_size
        if test_begin<0:
            test_begin=0 
 
    test_data=df.loc[index][test_begin:test_finish] 
    train_data=df.loc[index][:test_begin]
    if train_data.empty==False:
      last = df.loc[index][test_finish:]
    else:
      train_data=df.loc[index][test_finish:]
    #print(test_data)
    if i!=4:
      new = train_data.append(last)
    else:
      new = train_data 
    #print(new)

    sub_tree = tree.decision_tree_algorithm(new)
    tree.tree=sub_tree
    #print(test_data)

    rows,colums=test_data.values.shape
    y_hat = tree.predict(test_data.iloc[:,0:colums-1])
    y= test_data.iloc[:,-1]
    print()
    print('Fold: ',i+1)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
         print('Class Name: ',cls)
         print('Precision: ', precision(y_hat, y, cls))
         print('Recall: ', recall(y_hat, y, cls))
         print()

