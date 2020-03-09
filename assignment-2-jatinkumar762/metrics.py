import numpy as np

def accuracy(y_hat, y):
    """
    Function to calculate the accuracy

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the accuracy as float
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert(y_hat.size == y.size)
    count=0
    for i in range(y.size):
        if y_hat.iloc[i]==y.iloc[i]:
                count+=1  
    return ((count/y.size)*100)

def precision(y_hat, y, cls):
    """
    Function to calculate the precision

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the precision as float
    """
    assert(y_hat.size==y.size)
    count=0
    total=0
    for i in range(y_hat.size):
        if  y_hat.iloc[i]==cls:
            total+=1
    
    for i in range(y.size):
        if  y_hat.iloc[i]==cls and y_hat.iloc[i]==y.iloc[i]:
            count+=1
    
    if total!=0:
        return ((count/total)*100)
    else:
        return 0

def recall(y_hat, y, cls):
    """
    Function to calculate the recall

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    > cls: The class chosen
    Output:
    > Returns the recall as float
    """
    assert(y_hat.size==y.size)
    count=0
    total=0
    for i in range(y.size):
        if y.iloc[i]==cls:
            total+=1
    
    for i in range(y.size):
        if y.iloc[i]==cls and y.iloc[i]==y_hat.iloc[i]:
            count+=1
    
    if total!=0:
        return ((count/total)*100)
    else:
        return 0

def rmse(y_hat, y):
    """
    Function to calculate the root-mean-squared-error(rmse)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the rmse as float
    """
    total=0
    for i in range(y.size):
        total+=((y_hat.iloc[i] - y.iloc[i])**2)
    total/=y.size 

    return np.sqrt(total)

def mae(y_hat, y):
    """
    Function to calculate the mean-absolute-error(mae)

    Inputs:
    > y_hat: pd.Series of predictions
    > y: pd.Series of ground truth
    Output:
    > Returns the mae as float
    """
    total=0
    for i in range(y.size):
        total+=np.absolute(y_hat.iloc[i]-y.iloc[i])
    total/=y.size
    
    return total
