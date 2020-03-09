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

    #check only one class is there
    def check_purity(self,data):
        label_names=data[:,-1]
        if(len(np.unique(label_names))==1):
            return True
        return False

    #return the class whose count is maximum
    def data_classify(self,data):
        target=data[:,-1]
        unique,count = np.unique(target,return_counts=True)
        index=count.argmax()
        return unique[index]

    #find all potential splits
    def get_possible_splits(self,data):
        potential_splits={}
        rows,colums=data.shape
        for column_index in range(colums-1):
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
        target=train_data[:,-1]
        __,counts=np.unique(target,return_counts=True)
        probability=counts/counts.sum()
        entropy=sum(probability*-np.log2(probability))
        return entropy

    #calculate the overall weighted entropy
    def overall_entropy(self,data_below,data_above):
        n=len(data_below)+len(data_above)
        below_probability=len(data_below)/n
        above_probability=len(data_above)/n
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

    # returns the best column  to split with split point
    def best_split_reg(self,train_data,potential_splits):
        overall_metric=np.inf
        first_iteration=True
        for col in potential_splits:
            for val in potential_splits[col]:
                data_points_b,data_points_a=self.split_data(train_data,col,val)
                current_all_metric = self.overall_metric(data_points_b,data_points_a)
                if first_iteration or current_all_metric < overall_metric:
                    first_iteration=False
                    overall_metric = current_all_metric
                    best_split_column = col
                    best_split_value = val
        return best_split_column, best_split_value

    #split the data into 2 parts given a particular feature and split point
    def split_data_cat(self,train_data,split_column,split_value):
        column_values=train_data[:,split_column]
        return train_data[column_values==split_value]

    def best_split_cat(self,train_data,potential_splits):
        overall_entropy=np.inf
        total_entropy=0
        n=len(train_data)
        for col in potential_splits:
            total_entropy=0
            for val in potential_splits[col]:
                class_data=self.split_data_cat(train_data,col,val)
                #print(class_data)
                total_entropy+=(len(class_data)/n)*entropy(class_data[:,-1])
            #print(total_entropy)
            if(total_entropy<overall_entropy):
                best_split_column=col
                overall_entropy=total_entropy
        return best_split_column

    def best_split_catdis(self,train_data,potential_splits):
        overall_metric=np.inf
        total_metric=0
        n=len(train_data)
        for col in potential_splits:
            total_metric=0
            for val in potential_splits[0]:
                class_data=self.split_data_cat(train_data,col,val)
                if(len(class_data)<=1):
                    total_metric+=0
                else:
                    total_metric+=(len(class_data)/n)*np.var(class_data[:,-1],ddof=1)
            
            if(total_metric<overall_metric):
                best_split_column=col
                overall_metric=total_metric
        
        return best_split_column


    #work for real input and discrete data
    def decision_tree_algorithm(self, df,counter=0,max_depth=np.inf):

        if counter == 0:
            global column_names
            column_names = df.columns
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

            feature_name = column_names[split_column]
            question = "{} <= {}".format(feature_name,split_value)
            sub_tree = {question: []}
            true_answer = self.decision_tree_algorithm(data_below,counter,max_depth)
            false_answer = self.decision_tree_algorithm(data_above,counter,max_depth)

            if true_answer == false_answer:
                sub_tree = true_answer
            else:
                sub_tree[question].append(true_answer)
                sub_tree[question].append(false_answer)

        return sub_tree

    def create_leaf(self,data):
        label_column=data[:,-1]
        unique_classes,counts_unique_classes=np.unique(label_column,return_counts=True)
        index= counts_unique_classes.argmax()
        leaf=unique_classes[index]
        return leaf


    def regression_tree_algorithm(self,df,counter=0,min_samples=3,max_depth=np.inf):

        if counter == 0:
            global column_names
            column_names = df.columns
            data = df.values
        else:
            data = df
        
        if (self.check_purity(data)) or (len(data) < min_samples) or counter==max_depth:
            leaf=np.mean(data[:,-1])
            return leaf
        else:
            counter+=1

            potential_splits=self.get_possible_splits(data)
            split_column, split_value = self.best_split_reg(data,potential_splits)
            data_below, data_above = self.split_data(data,split_column,split_value)

            feature_name = column_names[split_column]
            question = "{} <= {}".format(feature_name,split_value)
            sub_tree = {question: []}
            true_answer = self.regression_tree_algorithm(data_below,counter,min_samples,max_depth)
            false_answer = self.regression_tree_algorithm(data_above,counter,min_samples,max_depth)

            if true_answer == false_answer:
                sub_tree = true_answer
            else:
                sub_tree[question].append(true_answer)
                sub_tree[question].append(false_answer)

            return sub_tree

    def catcat_tree_algorithm(self,df,counter=0,max_depth=np.inf):

        if counter == 0:
            global column_names
            column_names = df.columns
            data = df.values
        else:
            column_names = df.columns
            data = df.values

        if (self.check_purity(data)) or counter==max_depth:
            classify=self.data_classify(data)
            return classify
        else:
            counter+=1
            _,col=df.shape
            if col==2:
                potential_splits=self.cat_possible_splits(data)
                feature_name = column_names[0]
                split_keys=potential_splits[0]
                #print(potential_splits)
                #print(feature_name, split_keys)
                sub_tree = {}
                for i in range(len(split_keys)):
                    question = "{} $ {}".format(feature_name, split_keys[i])
                    sub_tree[question]=[]

               # print(len(split_keys))
                for i in range(len(split_keys)):
                    #cdata=df[df[feature_name]==split_keys[i]]
                    answer=self.data_classify(data)
                    question = "{} $ {}".format(feature_name, split_keys[i])
                    sub_tree[question].append(answer)

            else:
                potential_splits=self.cat_possible_splits(data)
                split_column=self.best_split_cat(data,potential_splits)
                feature_name = column_names[split_column]
                #print(split_column)
                split_keys=potential_splits[split_column]
                sub_tree = {}
                for i in split_keys:
                    question = "{} $ {}".format(feature_name, i)
                    sub_tree[question]=[]

                for i in range(len(split_keys)):
                    temp = df[df[feature_name]==split_keys[i]]
                    answer=self.catcat_tree_algorithm(temp.drop([feature_name],axis=1),counter,max_depth)
                    question = "{} $ {}".format(feature_name,split_keys[i])
                    sub_tree[question].append(answer)

            return sub_tree

    def catdis_tree_algorithm(self,df,counter=0,max_depth=np.inf):

        if counter == 0:
            global column_names
            column_names = df.columns
            data = df.values
        else:
            column_names = df.columns
            data = df.values

        if (self.check_purity(data)) or counter==max_depth:
            leaf=np.mean(data[:,-1])
            return leaf
        else:
            counter+=1
            
            _,col=df.shape
            if col==2:
                potential_splits=self.cat_possible_splits(data)
                feature_name = column_names[0]
                split_keys=potential_splits[0]
                sub_tree = {}
                for i in split_keys:
                    question = "{} $ {}".format(feature_name, i)
                    sub_tree[question]=[]

                for i in range(len(split_keys)):
                    cdata=df[df[feature_name]==split_keys[i]]
                    answer=np.mean(cdata[:,-1])
                    question = "{} $ {}".format(feature_name, split_keys[i])
                    sub_tree[question].append(answer)

            else:
                potential_splits=self.cat_possible_splits(data)
                split_column=self.best_split_catdis(data,potential_splits)
                feature_name = column_names[split_column]
                #print(split_column)
                split_keys=potential_splits[split_column]
                sub_tree = {}
                for i in split_keys:
                    question = "{} $ {}".format(feature_name, i)
                    sub_tree[question]=[]

                for i in range(len(split_keys)):
                    temp = df[df[feature_name]==split_keys[i]]
                    answer=self.catdis_tree_algorithm(temp.drop([feature_name],axis=1),counter,max_depth)
                    question = "{} $ {}".format(feature_name,split_keys[i])
                    sub_tree[question].append(answer)
            
            return sub_tree


    def fit(self, X, y):
        """
        Function to train and construct the decision tree
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """

        self.input = str(X.dtypes[0])
        self.output = str(y.dtypes)

        X['label'] = y
        X.columns = ['A', 'B', 'C', 'D', 'E', 'Y']

        if self.input == "category" and self.output == "category":
                self.tree = self.catcat_tree_algorithm(X)
        elif self.input == "category" and self.output != "category":
                self.tree = self.catdis_tree_algorithm(X)
        elif self.input != "category" and self.output == "category":
                self.tree = self.decision_tree_algorithm(X)
        else:
                self.tree = self.regression_tree_algorithm(X)

        

    #given an unknown test example assign it's class
    def classify_example(self,example, tree):
        question = list(tree.keys())[0]
        feature_name, comparison_operator, value = question.split(" ")
        if comparison_operator == "<=":
            if example[feature_name] <= float(value):
                answer = tree[question][0]
            else:
                answer = tree[question][1]
        else:
            if str(example[feature_name]) == value:
                answer = tree[question][0]
            else:
                answer = tree[question][1]
                
        if not isinstance(answer, dict):
            return answer
    
        else:
            residual_tree = answer
            return self.classify_example(example, residual_tree)

    #given an unknown test example assign it's class
    def cat_classify_example(self,example, tree):
        for ques in list(tree.keys()):
           feature_name, comparison_operator, value = ques.split(" ")
           if comparison_operator == "$":
               if(str(example[feature_name])) == str(value):
                   answer = tree[ques][0]
                
        if not isinstance(answer, dict):
            return answer
        else:
            residual_tree = answer
            return self.cat_classify_example(example, residual_tree)


    def predict(self, X):
        """
        Funtion to run the decision tree on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        if (self.input=="category" and self.output=="category") or (self.input=="category" and self.output!="category"):
          return X.apply(self.cat_classify_example, axis=1, args=(self.tree,))
        else:
          return X.apply(self.classify_example, axis=1, args=(self.tree,))


    def plot(self):
        """
        Function to plot the tree

        Output Example:
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        pprint(self.tree)

    #train test split in 70/30 ratio
    def train_test_split(self,df):
        index=df.index.tolist()
        np.random.shuffle(index)
        len,_=df.values.shape
        train_size=int(len*0.7)
        train_data=df.loc[index][0:train_size]
        test_data=df.loc[index][train_size:]  
        return train_data,test_data