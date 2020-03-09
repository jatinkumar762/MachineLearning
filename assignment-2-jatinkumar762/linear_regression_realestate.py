import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import *


df=pd.read_excel("Real estate valuation data set.xlsx",names=['No','tran_date','age','distance_mrt','stores','lat','long','price'])
df=df.drop('No',axis=1)

index=df.index.tolist()
len,_=df.values.shape

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

    if i!=4:
      new = train_data.append(last)
    else:
      new = train_data 
    
    # print(new)
    # print()
    # print(test_data)

    _,col = new.shape
    X = new.iloc[:,0:col-1]
    y = new.iloc[:,-1]
    fit_intercept = True
    print('Fold: ',i+1)
    print('fit_intercept: ',fit_intercept)
    LR = LinearRegression(fit_intercept=fit_intercept)
    LR.fit(X, y)
    y_hat = pd.Series(LR.predict(X))
    LR.y_hat = y_hat
    LR.plot()
    print('learnt coefficients: ',LR.theta)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    print()