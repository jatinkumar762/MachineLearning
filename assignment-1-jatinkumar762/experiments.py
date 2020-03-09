
import pandas as pd
import numpy as np
import timeit
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from pprint import pprint 

np.random.seed(42)
num_average_time = 100

# Learn DTs 
# ...
# 
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions

# Test case 1
# Real Input and Real Output

N = 50
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

try:
    for criteria in ['information_gain', 'gini_index']:
        tree = DecisionTree(criterion=criteria,max_depth=10) #Split based on Inf. Gain
        start = timeit.default_timer()
        tree.fit(X, y)
        stop = timeit.default_timer()
        print('Real Input and Real Output Time - Build Tree: ', stop - start)
        start = timeit.default_timer()
        y_hat = tree.predict(X)
        stop = timeit.default_timer()
        print('Real Input and Real Output Time - Predict Tree: ', stop - start)
except:
    pass

# Test case 2
# Real Input and Discrete Output

X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

try:
    for criteria in ['information_gain', 'gini_index']:
        tree = DecisionTree(criterion=criteria,max_depth=10) #Split based on Inf. Gain
        start = timeit.default_timer()
        tree.fit(X, y)
        stop = timeit.default_timer()
        print('Real Input and Discrete Output Time - Build Tree: ', stop - start)
        start = timeit.default_timer()
        y_hat = tree.predict(X)
        stop = timeit.default_timer()
        print('Real Input and Discrete Output Time - Predict Tree: ', stop - start)
except:
    pass

# Test case 3
# Discrete Input and Discrete Output

N = 30
P = 5
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randint(P, size = N), dtype="category")


try:
    for criteria in ['information_gain']:
        tree = DecisionTree(criterion=criteria,max_depth=10) #Split based on Inf. Gain
        tree.output="category"
        tree.input="category"

        #df=pd.read_csv("tennis.csv")
        #subtree=tree.catcat_tree_algorithm(df)
        #tree.tree=subtree
        #pprint(subtree)
        #y_hat = tree.predict(df)
        #y=df.iloc[:,-1]
        #print('Accuracy: ', accuracy(y_hat, y))
        #for cls in y.unique():
        #    print(cls)
        #    print('Precision: ', precision(y_hat, y, cls))
        #    print('Recall: ', recall(y_hat, y, cls))

        start = timeit.default_timer()
        tree.fit(X, y)
        stop = timeit.default_timer()
        print('Discrete Input and Discrete Output - Build Tree: ', stop - start)
        start = timeit.default_timer()
        y_hat = tree.predict(X)
        stop = timeit.default_timer()
        print('Discrete Input and Discrete Output - Predict Tree: ', stop - start)
        #y=df.iloc[:,-1]
        #tree.plot()
except:
    pass

# Test case 4
# Discrete Input and Real Output

N = 30
P = 5
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

try:
    tree = DecisionTree(criterion='information_gain',max_depth=10) #Split based on Inf. Gain
    start = timeit.default_timer()
    tree.fit(X, y)
    stop = timeit.default_timer()
    print('Discrete Input and Real Output - Build Tree: ', stop - start)
    start = timeit.default_timer()
    y_hat = tree.predict(X)
    stop = timeit.default_timer()
    print('Discrete Input and Real Output - Predict Tree: ', stop - start)
except:
    pass



N = 100
P = 5
X = pd.DataFrame({i:pd.Series(np.random.randint(P, size = N), dtype="category") for i in range(5)})
y = pd.Series(np.random.randn(N))

try:
    tree = DecisionTree(criterion='information_gain',max_depth=10) #Split based on Inf. Gain
    start = timeit.default_timer()
    tree.fit(X, y)
    stop = timeit.default_timer()
    print('Discrete Input and Real Output - Build Tree: ', stop - start)
    start = timeit.default_timer()
    y_hat = tree.predict(X)
    stop = timeit.default_timer()
    print('Discrete Input and Real Output - Predict Tree: ', stop - start)
except:
    pass


line, =plt.plot([30,100],[0.12184499999999998, 0.2225138000000002], label='Build Tree')
plt.plot([30,100],[0.013385800000000003, 0.016191500000000136], label='Predict Tree')
plt.title('Comparison for different input size data')
plt.xlabel('Rows in Data')
plt.ylabel('Time in sec.')
#line.set_label('Label via method')
plt.legend()
plt.show()


    
