"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index
from pprint import pprint

np.random.seed(42)

class DecisionTree():
    def __init__(self, criterion, max_depth):
        """
        Put all infromation to initialize your tree here.
        Inputs:
        > criterion : {"information_gain", "gini_index"}
        """
        self.criterion = criterion
        self.max_depth= max_depth
        self.tree= {}
        self.input = ""
        self.output = ""
        self.splitColumn = None

    #check only one class is there
    def check_purity(self,data):
        label_names=data[:,-2]
        if(len(np.unique(label_names))==1):
            return True
        return False

    #return the class whose count is maximum
    def data_classify(self,data):
        target=data[:,-2]
        unique,count = np.unique(target,return_counts=True)
        index=count.argmax()
        return unique[index]

    #find all potential splits
    def get_possible_splits(self,data):
        potential_splits={}
        rows,colums=data.shape
        for column_index in range(colums-2):
            potential_splits[column_index]=[]
            unique_values=np.unique(data[:,column_index])  #eturn the sorted unique elements of the array
            for i in range(1,len(unique_values)):
                cv=unique_values[i]
                pv=unique_values[i-1]
                potential_split_point=(float(cv)+float(pv))/2
                potential_splits[column_index].append(potential_split_point)        
        return potential_splits

    #Category potential splits
    def cat_possible_splits(self,data):
        potential_splits={}
        rows,colums=data.shape
        for col_index in range(colums-1):
            potential_splits[col_index]=np.unique(data[:,col_index])
        return potential_splits

    #split the data into 2 parts given a particular feature and split point
    def split_data(self,train_data,split_column,split_value):
        column_values=train_data[:,split_column]
        data_below=train_data[column_values<split_value]
        data_above=train_data[column_values>=split_value]
    
        return data_below,data_above

    #calculate the entropy of the data
    def entropy_cal(self,train_data):

        # target=train_data[:,-2]
        # R,counts=np.unique(target,return_counts=True)
        # probability=counts/counts.sum()
        # entropy=sum(probability*-np.log2(probability))

        temp = train_data.copy()
        row,col = temp.shape
        unq = np.unique(temp[:,-2])
        total_sum = sum(temp[:,-1])
        data = []
        entropy = 0
        for u in unq:
            data.clear()
            for r in range(row):
                if temp[r][2] == u:
                    data.append(temp[r][-1])
            p = sum(data)/total_sum
            entropy += (p*-np.log2(p))
        return entropy

    #calculate the overall weighted entropy
    def overall_entropy(self,data_below,data_above):
        n=sum(data_below[:,-1])+sum(data_above[:,-1])
        below_probability=sum(data_below[:,-1])/n
        above_probability=len(data_above[:,-1])/n
        overall=below_probability*(self.entropy_cal(data_below))+above_probability*(self.entropy_cal(data_above))
        return overall

    # returns the best column  to split with split point
    def best_split(self,train_data,potential_splits):
        overall_entropy=np.inf
        for col in potential_splits:
            for val in potential_splits[col]:
                data_points_b,data_points_a=self.split_data(train_data,col,val)
                current_all_entropy = self.overall_entropy(data_points_b,data_points_a)
                if current_all_entropy < overall_entropy:
                    overall_entropy = current_all_entropy
                    best_column = col
                    best_value = val
        
        return best_column,best_value

    # this function return the total RMS value
    def overall_metric(self,data_below,data_above):
        #print(data_below[:,-1])
        #print(data_above[:,-1])
        #total_variance=np.var(data_below[:,-1],ddof=1)+np.var(data_above[:,-1],ddof=1)
        #print(np.var(data_below[:,-1],ddof=1))
        #print(np.var(data_above[:,-1],ddof=1))
        mb=np.mean(data_below)
        ma=np.mean(data_above)
        mse=np.mean((data_below-mb)**2)+np.mean((data_above-ma))
        return float(mse)


    #work for real input and discrete data
    def decision_tree_algorithm(self, df,counter=0,max_depth=1):

        if counter == 0:
            column_names = df.columns.tolist()
            data = df.values
        else:
            data = df
        
        if (self.check_purity(data)) or counter>=max_depth:
            catg=self.data_classify(data)
            return catg
        else:
            counter+=1

            potential_splits=self.get_possible_splits(data)
            split_column, split_value = self.best_split(data,potential_splits)
            data_below, data_above = self.split_data(data,split_column,split_value)

            #print(split_column)
            #print(column_names[split_column])
            feature_name = column_names[split_column]
            #print(feature_name)
            question = "{} <= {}".format(feature_name,split_value)
            sub_tree = {question: []}
            #print(sub_tree)
            true_answer = self.decision_tree_algorithm(data_below,counter,max_depth)
            false_answer = self.decision_tree_algorithm(data_above,counter,max_depth)

            # if true_answer == false_answer:
            #     sub_tree[question] = true_answer
            # else:
            sub_tree[question].append(true_answer)
            sub_tree[question].append(false_answer)

        #print(sub_tree)
        return sub_tree,feature_name

    def create_leaf(self,data):
        label_column=data[:,-1]
        unique_classes,counts_unique_classes=np.unique(label_column,return_counts=True)
        index= counts_unique_classes.argmax()
        leaf=unique_classes[index]
        return leaf

    def fit(self, X, y, W):

        X['label'] = y
        X['Weight'] = W
        #X.columns = ['A', 'B', 'label', 'Weight']
        if self.input != "category" and self.output == "category":
            self.tree,self.splitColumn = self.decision_tree_algorithm(X)

        

    #given an unknown test example assign it's class
    def classify_example(self,example, tree):
        example = example.to_numpy()
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")
        if comparison_operator == "<=":
            if example[0] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        return answer

    def predict(self, X):

        return X.apply(self.classify_example, axis=1, args=(self.tree,))


    def plot(self):
        print(self.tree)

    #train test split in 70/30 ratio
    def train_test_split(self,df):
        index=df.index.tolist()
        np.random.shuffle(index)
        len,_=df.values.shape
        train_size=int(len*0.7)
        train_data=df.loc[index][0:train_size]
        test_data=df.loc[index][train_size:]  
        return train_data,test_data