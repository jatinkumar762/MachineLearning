
def entropy(Y):
    """
    Function to calculate the entropy 

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the entropy as a float
    """   
    __,counts=np.unique(Y,return_counts=True)
    probability=counts/counts.sum()
    entropy=sum(probability*-np.log2(probability))
    
    return entropy

def gini_index(Y):
    """
    Function to calculate the gini index

    Inputs:
    > Y: pd.Series of Labels
    Outpus:
    > Returns the gini index as a float
    """
    total=Y.size
    unique=Y.value_counts().to_dict()
    gini=1
    for k,c in unique.items():
        gini-=(c/total)**2

    return gini

def information_gain(Y, attr):
    """
    Function to calculate the information gain
    
    Inputs:
    > Y: pd.Series of Labels
    > attr: pd.Series of attribute at which the gain should be calculated
    Outputs:
    > Return the information gain as a float
    """
    total_entropy = entropy(Y)
    ent=0
    w_avg=0
    labels = attr.unique()
    tmp_list = []
    for i in range(len(labels)):
            temp_list.clear()
            for j in range(attr.size):
                if labels[i] == attr[j]:
                    tmp_list.append(Y[j])
            x = pd.Series(temp_list)
            ent = entropy(x)
            w_avg=(-1)*((x.size/attr.size)*ent)
            total_entropy-=w_avg

    return total_entropy
