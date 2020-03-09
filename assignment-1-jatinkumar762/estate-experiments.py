
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *


from sklearn import tree as sktree
from sklearn import metrics
from sklearn.model_selection import train_test_split
import numpy as np
from pprint import pprint

np.random.seed(42)

# Read real-estate data set
# ...
# 
tree = DecisionTree(criterion='information_gain',max_depth=10) #Split based on Inf. Gain
tree.output="discrete"
tree.input="discrete"
df=pd.read_excel("Real estate valuation data set.xlsx",names=['No','tran_date','age','distance_mrt','stores','lat','long','price'])
df=df.drop('No',axis=1)
train_data,test_data=tree.train_test_split(df)
sub_tree = tree.regression_tree_algorithm(df)
print(sub_tree)
tree.tree=sub_tree
rows,colums=test_data.values.shape
y_hat = tree.predict(test_data.iloc[:,0:colums-1])
y= test_data.iloc[:,-1]
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))

#N = 30
#P = 5
#X = pd.DataFrame(np.random.randn(N, P))
#y = pd.Series(np.random.randn(N))

#tree.fit(X, y)
#y_hat = tree.predict(X)
#tree.plot()
#print('Criteria :', criteria)
#print('RMSE: ', rmse(y_hat, y))
#print('MAE: ', mae(y_hat, y))


data=df.values
target_data=data[:,-1]
features=df.iloc[:,1:6]
feature_data=features.values
X_train, X_test, y_train, y_test = train_test_split(feature_data, target_data, test_size=0.30, random_state=15)
c=sktree.DecisionTreeRegressor(max_depth=10, min_samples_split=3)
c.fit(X_train,y_train)
y_predict=c.predict(X_test)
print('**Skikit learn Data**')
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))


