# Single-output variation of the AR-XGBOOST algorithm.

import os
import pandas as pd
import numpy as np
import numpy.matlib
from dateutil.parser import parse
import matplotlib.pyplot as mp
import time 
import statsmodels.api as sm


# start_time = time.time()  # Used to measure the runtime starting from this point


path = 'C:\\Users\\sam37\\Desktop\\Master-Thesis\\Data' # Directory to access data
os.chdir(path)


# Dataset containing prices
data = pd.read_csv('3571from1995ns.csv') 

# Dataset containing fundamentals 
finstat = pd.read_csv('finstat.csv')  

finstat = finstat.rename(columns={'dvpspq':'div','epspxq':'eps',
                                  'oancfy':'cf','oibdpq':'opin',
                                  'saleq':'sales'})


# Explore lentghs of time-series for each stock and keep stocks with 
# full time-series)
len_data = data.groupby('TICKER').apply(len)

len_fin = finstat.groupby('tic').apply(len)

valid_tic = len_data[len_data==max(len_data)].index

data = data[data['TICKER'].isin(valid_tic)]

finstat = finstat[finstat['tic'].isin(valid_tic)]

finstat = finstat.rename(columns={
    'dvpspq':'div','epspxq':'eps','oancfy':'cf','oibdpq':'opin','saleq':'sales'
    })


# Explore start and end of datasets
data['date'] = pd.to_datetime(data['date'], format='%Y/%m/%d')

finstat['datadate'] = pd.to_datetime(finstat['datadate'], format='%Y/%m/%d')

maxfinstat = finstat.groupby('tic').apply(lambda x: max(x['datadate']))

maxdata = data.groupby('TICKER').apply(lambda x: max(x['date']))

minfinstat = finstat.groupby('tic').apply(lambda x: min(x['datadate']))

mindata = data.groupby('TICKER').apply(lambda x: min(x['date']))


# Match dates for data about prices with data about fundamentals
data = data[(data['date']>=max(minfinstat)) & (data['date']<(
    min(maxfinstat) + pd.DateOffset(months=4)))]


# Book value (in millions)
finstat['bookv'] = finstat['atq'] - finstat['ltq']  # assets - liabilities
# Market value
data['mktv'] = data['PRC']*data['SHROUT']  # price * (outstanding shares)


# Transform quarterly series to daily series and add them to the 'data' 
# DataFrame
finstat = finstat.rename(columns={'datadate':'date', 'tic':'TICKER'})

temp_finstat = finstat[['date','TICKER','bookv','div','eps','cf','opin','sales']]

data = data.merge(temp_finstat,
                  on=['TICKER', 'date'], how='left').fillna(method='ffill')

# Select the stock we want to analyze
data = data[data['TICKER']=='AAPL']

# Build financial ratios dataframe
fratios = pd.DataFrame()
fratios['ticker'] = data['TICKER']
fratios['bm'] = data['bookv']/data['mktv']
fratios['dy'] = data['div']/data['PRC']
fratios['pe'] = data['PRC']/data['eps']
fratios['poin'] = data['PRC']/data['opin']
fratios['pcf'] = data['PRC']/data['cf']
fratios['psales'] = data['PRC']/data['sales']


# Calculate returns
return_span = 4
returns = (np.array(data['PRC'][return_span:])-np.array(
    data['PRC'][:-(return_span)]))/np.array(
        data['PRC'][:-(return_span)])
        
''' lag_inpupt creates the vector of dependent variables from  'y' and 
the matrix of regressors from 'x'. n_x_lags is the number of lags to 
create for the variables in x. lag_length is the length of lags. 
To include an intercept set it to 1. The default value is 0 which indicates
no intercept. '''        
def lag_input(y, x, n_x_lags, lag_length, intercept=0):
    y = y[n_x_lags*lag_length:]
    temp = np.ones((len(y),1))
    for i in range(n_x_lags-1,-1,-1):
        temp = np.concatenate(
            (temp, x[i*lag_length:-(n_x_lags-i)*lag_length,:]),
            axis=1)
    if intercept == 0:
        return y, temp[:,1:]
    else:
        return y, temp

# Prepare inputs
input_x = np.concatenate(
    (returns[:,np.newaxis], np.array(fratios.iloc[return_span:, 1:])), axis=1
    )


# Obtain inputs for analysis
y, x = lag_input(returns, input_x, 2, 5)
        

# Separate the input data into train and test
data_T = len(x)
x_train = x[0:data_T-1000,:]
y_train = y[0:data_T-1000]
x_test = x[data_T-1000:-500,:]
y_test = y[data_T-1000:-500]


# Regress y on x to find a first estimate of beta, beta_0
n = len(y_train)
m = np.shape(x_train)[1]
model = sm.OLS(y_train,x_train)
leaff = model.fit()
beta_0 = leaff.params
beta_0 = np.ones((n,m))*beta_0.T


# # **Uncomment this section to work with realized volatility**
# # We first calculate log returns to then calculate realized volatilities
# log_returns = np.log(np.array(data['PRC'][1:])/np.array(data['PRC'][0:-1]))

# # This function calculates the realized volatility for 'log_returns' over
# # the time horizon corrisponding to 'span'.
# def real_vol(log_returns,span):
#     kernel = np.ones(span)
#     vol = np.sqrt(np.convolve(np.square(log_returns), kernel, mode='valid'))
#     return vol[:,np.newaxis]


# # We calculate realized volatilities over 5,10 and 20 days.
# vol5 = real_vol(log_returns,5)
# vol10 = real_vol(log_returns,10)
# vol20 = real_vol(log_returns,20)

# # The target variable 'y' consists of 5 days realized volatilities.
# y = vol5[20:,0] 

# # We create the matrix 'x' of regressors with 2 lags of the 6 financial ratios
# # and the 5,10 and 20 days realized volatilities.
# y, x = lag_input(vol5, np.array(fratios.iloc[5:,1:]), 2, 5)
# y = y[10:]
# x = x[10:,:]
# x = np.concatenate((x, vol5[15:-5], vol10[10:-5], vol20[:-5]), axis=1)

# # We separate the data into training and test sets.
# data_T = len(x)
# x_train = x[0:data_T-1000,:]
# y_train = y[0:data_T-1000]
# x_test = x[data_T-1000:-500,:]
# y_test = y[data_T-1000:-500]

# Function that calculates the RMSPE. 'y' is the real value, 'y_hat'
# is the estimate.
def rmspe(y,y_hat):
    temp = np.square((y-y_hat)/y)
    idfinite = np.isfinite(temp)
    rmspe = np.sqrt(np.mean(temp[idfinite]))
    return rmspe


# Functions that calculate the gradient and the hessian for each data point.
# Both gradients and hessians are calculated from the mean squared error.
def gradient(y,x,beta):
    grad = -(y-np.diagonal(x@beta.T))
    return np.expand_dims(grad,axis=1)
def hessian(x):
    hess = 1
    return hess


'''To split a leaf we need to compare the possible scores, one for each possible split.
In order to do so, we need to know what the subset of 'x' for the current leaf is. The argument
'x' in this function already contains the current subset.
Then, calculate the score using the current subset, the subset resulting from split 1
and the subset resulting from split 2. 'split_val' is the value of the split, 
'split_var' is the feature that determines the split, grad and hess are the gradients
and hessians, 'y' is the vector of target variables corresponding to the subset 
of the current leaf. It returns the score and the two subsets resulting from the split.'''
def score(split_val,split_var,grad,hess,llambda,gamma,x,y):
    split2 = x[:,split_var]>split_val
    split1 = x[:,split_var]<=split_val
    L = -1/2*grad.T@x@np.linalg.pinv(x.T@x+llambda*np.eye(np.shape(x)[1]))@x.T@grad
    L1 = -1/2*grad[split1].T@x[split1]@np.linalg.pinv(x[split1].T@x[split1]+llambda*np.eye(np.shape(x[split1])[1]))@x[split1].T@grad[split1]
    L2 = -1/2*grad[split2].T@x[split2]@np.linalg.pinv(x[split2].T@x[split2]+llambda*np.eye(np.shape(x[split2])[1]))@x[split2].T@grad[split2]
    sscore = L - L1 - L2 - gamma
    return sscore, split1, split2


# We create a 'leaf' class. The initial argument is the subset of data (the indexes
# corresponding to the data) within the leaf.
class leaf():
   
    
    def __init__(self,subset):
        # 'split_val' and 'split_var' store the value and variable of the split
        # and they are used to determine the path of new data when it is fed to 
        # the algorithm.
        self.split_val = []
        self.split_var = []
        # 'lr' stores 'l' if it is a left split (<=split_val) or 'r' if it is
        # a right split (>split_val)
        self.lr = []
        # Subset used to calculate the score, beta and y_hat (it is a set of indexes
        # of the data points that belong to the current 'x'. All these data points are 
        # assigned to the same beta)
        self.subset = subset
        # 'beta' stores the vector of beta coefficients which can be used to calculate
        # the forecast 'y_hat'.
        self.beta = []
        # 'endnode' stores 'yes' if it is an end node, i.e. a leaf.
        self.endnode = []
        # 'id_next_leaf' indicates the index of the left node of the split
        # in the layer below.'id_next_leaf + 1' is the right node of the split.
        self.id_next_leaf = [] # index of the left leaf in the layer below
    
    
    # The approximate split finding algorithm is carried out by the 
    # 'find_split' function. The inputs include the matrix of regressors 'x',
    # the vector of target variables 'y' and the vector of coefficients 'beta',
    # all associated to the parent node. This function returns the value 
    # of the feature that determines the split, its index, the indexes of the left 
    # region and the indexes of the right region.
    def find_splitt(self,llambda,gamma,x,y,beta):
        gainn = 0
        # 'sub1' and 'sub2' store the resulting subsets of the split.
        sub1 = []
        sub2 = []
        # If 'x' has more than 200 data points we proceed with the approximate split
        # finding algorithm. Otherwise, we employ the exact greedy split finding algorithm.
        if len(x)>200:
            # Calculate the gradient, the 'xg' and 'xhx' terms for the dataset
            # of the parent node.
            grad = gradient(y,x,beta)
            xg = x*grad
            xhx = x.T@x
            # Calculate the loss corresponding to the parent node.
            L = -1/2*np.sum(xg,axis=0)@np.linalg.pinv(xhx+llambda*np.eye(np.shape(x)[1]))@np.sum(xg,axis=0)
            # In this 'for loop' we caclulate the 'gain' for the split corresponding
            # to each of the 200 bins for each feature.
            for i in range(0,x.shape[1]): # This loop runs through the features
                # Create the 200 percentiles for feature 'i'
                steps = np.linspace(100/200,100,num=200)
                percs = np.percentile(x[:,i],steps)
                # Create the bins for the 'xg' and 'xhx' terms and a bin,
                # 'bins_id', that stores the index of the corresponding data points
                xg_bin = []
                xhx_bin = []
                bins_id = []
                # Because the first bin corresponds to a left-sided region and not 
                # an interval, this step takes place before the next 'for loop'.
                # We first find the indexes corresponding the left-sided region and 
                # store the 'xg' and 'xhx' terms corresponding to those indexes.
                bins_id.append(np.where(x[:,i]<=percs[0]))
                xg_bin.append(np.sum(xg[bins_id[0][0]],axis=0))
                xhx_bin.append(x[bins_id[0][0],:].T@x[bins_id[0][0],:])
                # This loop runs through the remaining percentiles to fill the bins
                for k in range(1,200): 
                    bins_id.append(np.where(np.logical_and(x[:,i]>percs[k-1], x[:,i]<=percs[k])))
                    xg_bin.append(np.sum(xg[bins_id[k][0]],axis=0))
                    xhx_bin.append(x[bins_id[k][0],:].T@x[bins_id[k][0],:])
                # We turn the lists into numpy arrays
                xg_bin = np.array(xg_bin)
                xhx_bin = np.array(xhx_bin)
                # We initialize the 'xg' and 'xhx' terms for the left region of the split
                leftxg = 0
                leftxhx = 0
                # In this 'for loop' we add the 'xg' and 'xhx' terms of bin 'j' 
                # to the previous totals corresponding to the left region. In other words,
                # in each iteration we calculate the cumulative 'xg' and 'xhx' of the left
                # and right regions for a certain split.
                for j in range(0,200-1):
                    # The new 'xg' and 'xhx' terms for the left region are equal to 
                    # the previous left-totals plus the values in bin 'j'
                    leftxg = leftxg + xg_bin[j]
                    leftxhx = leftxhx + xhx_bin[j,:,:]
                    # The new 'xg' and 'xhx' terms for the right region are equal to 
                    # the overall totals minus the current left totals                     
                    rxg = np.sum(xg,axis=0) - leftxg
                    rxhx = xhx - leftxhx
                    # Calculate the losses corresponding to the current left 
                    # and right regions. Then calculate the gain defined by 'scoree'
                    Ll = -1/2*leftxg@np.linalg.pinv(leftxhx+llambda*np.eye(np.shape(x)[1]))@leftxg
                    Lr = -1/2*rxg@np.linalg.pinv(rxhx+llambda*np.eye(np.shape(x)[1]))@rxg
                    scoree = L - Ll - Lr - gamma
                    # If the current gain is larger than the previous largest gain,
                    # we store the indexes corresponding to the current two regions into
                    # 'sub1' and 'sub2'. We also store the feature and the value of
                    # the split into 'split_var_temp' and 'split_val_temp' respectively.
                    if scoree>gainn:
                        left_x = np.squeeze(np.hstack(bins_id[:j+1]))
                        right_x = np.squeeze(np.hstack(bins_id[j+1:]))
                        gainn = scoree
                        split_val_temp = percs[j]
                        split_var_temp = i
                        sub1 = left_x
                        sub2 = right_x
        else:
            # In this 'else' section is the exact greedy algorithm.
            # For each feature we first sort the values.
            for i in range(0,x.shape[1]):
                sorted_x = np.sort(x[:,i])
                # For each of the sorted values we calculate the average between the next 
                # value and itself. The average is then used to determine the split and
                # calculate the gain.
                for j in range(0,x.shape[0]-1):
                    # grad and hess are inputs for gradients and hessians.
                    # Inputs are passed for the entire subset of the current leaf.
                    # Then, inside 'score', smaller subsets will be assigned to the 
                    # corresponding splits.
                    grad = gradient(y,x,beta)
                    hess = hessian(x)
                    value = (sorted_x[j+1]+sorted_x[j])/2
                    scoree, sub1_temp, sub2_temp = score(value,i,grad,hess,llambda,gamma,x,y)
                    # If the gain is larger than the previous largest gain, the value 
                    # of the feature, the index of the feature, the indexes of the left 
                    # region and the indexes of the right region are stored into
                    # 'split_val_temp','split_var_temp','sub1' and 'sub2' respectively.
                    if scoree>gainn:
                        gainn = scoree
                        split_val_temp = value
                        split_var_temp = i
                        sub1 = sub1_temp
                        sub2 = sub2_temp
        # If the largest possible gain is not larger than 0, it means no further split
        # is beneficial. In this case, we stop the splitting process and write 'stop'
        # into 'split_val_temp'. 'stop' will signal the current node is an end node.                
        if gainn<=0:
            split_val_temp = ['stop']
            split_var_temp = []
        return split_val_temp, split_var_temp, sub1, sub2    
   
    
    # Function that calculates the beta coefficients for the current leaf. It takes 
    # the entire matrix of regressors 'x' and vector of target variables 'y' as input.
    # In addition, it takes the entire matrix of beta coefficients from the previous 
    # tree as input.
    def calculate_beta(self,llambda,x_tot,y_tot,prev_beta_tot): 
        # We first obtain the subsets of 'x' and 'y' corresponding to the current leaf.
        sub = self.subset
        x = x_tot[sub]
        y = y_tot[sub]
        # We then obtain the subset of the previous beta coefficients corresponding
        # to the current leaf.
        prev_beta = prev_beta_tot[sub]
        # Finally we calculate the new beta
        beta1 = -np.linalg.pinv(x.T@x+llambda*np.eye(np.shape(x)[1]))@(x.T@gradient(y,x,prev_beta))
        self.beta = beta1


# We create a 'tree' class. The arguments are 'depth', the number of layers in the tree
# and 'prev_beta', the set of coefficients from the previous tree. 'previous_beta' is 
# initialized with beta_0, which was calculated with a simple linear regression.
class tree():
    
    
    def __init__(self,depth,prev_beta):
        self.depth = depth # Number of layers
        self.leaves = [[leaf([])]] # A list of 'leaf' objects that form the tree.
        self.prev_beta = prev_beta # Beta coefficients of the previous tree
        self.beta_hat = [] # Beta coefficients of the current tree
  
    
    # Method that builds the tree. It takes the matrix of regressors 'x',
    # the vector of target variables 'y' and the hyperparameters lambda
    # and gamma as arguments.
    def build_tree(self,x,y,llambda,gamma):
        # We create the initial node with the indexes of the full dataset
        # in 'subset' and calculate the set of coefficients corresponding
        # to the full dataset using the coefficients of the last tree.
        self.leaves[0][0].subset = np.arange(0,len(x))
        self.leaves[0][0].calculate_beta(llambda,x,y,self.prev_beta)
        # We create a set of leaves for each layer.
        for i in range(0,self.depth-1): 
            # Create a new empty layer of leaves.
            self.leaves.append([])     
            # For each of the leaves in the layer above the one we just created,
            # we create two leaves whenever a new optimal split is possible.
            for j in range(0,len(self.leaves[i])):
                # First, we access the the indexes of the subset of node 'j' 
                # in layer 'i'.
                sub = self.leaves[i][j].subset
                # If the node contains less than 7 data points, we mark the 
                # node as an end node.
                if len(sub)<7:
                    self.leaves[i][j].endnode = 'yes'
                    continue
                # We use the indexes in 'subset' to retrieve the corresponding
                # values of the features in 'x'.
                sub_x = x[sub]
                # We use the same indexes to retrieve the corresponding 
                # coefficients from the previous tree.
                beta_prev_sub = self.prev_beta[sub]
                # We retrieve the target values in 'y' corresponding to the subset.
                y_sub = y[sub]
                # Find the optimal split, if possible.
                splitt1,splitt2, sub1, sub2 = self.leaves[i][j].find_splitt(llambda,gamma,sub_x,y_sub,beta_prev_sub) 
                # If no optimal split is possible, the current node becomes an 
                # end node.
                if splitt1 == ['stop']:
                    self.leaves[i][j].endnode = 'yes'
                    continue
                # If an optimal split is possible, we create two new nodes in 
                # layer i+1 and include all information regarding the split 
                # into the current node and into the two new nodes.
                self.leaves[i+1].append(leaf(sub[sub1]))
                self.leaves[i+1].append(leaf(sub[sub2]))              
                self.leaves[i][j].split_val = splitt1
                self.leaves[i][j].split_var = splitt2                
                self.leaves[i+1][-1].lr = ['r']
                self.leaves[i+1][-2].lr = ['l']
                self.leaves[i+1][-1].calculate_beta(llambda,x,y,self.prev_beta)
                self.leaves[i+1][-2].calculate_beta(llambda,x,y,self.prev_beta)
                self.leaves[i][j].id_next_leaf = len(self.leaves[i+1])-2  
                       
    
    # Method that creates the 'beta_hat' matrix of the current tree.
    # Each row of the matrix is a set of coefficients for the corresponding
    # row of the input dataset 'x'. It is used to create a matrix of coefficients
    # where each row corresponds to a row in the training dataset 'x_train'.
    def estimate_betas(self,x):
        beta_temp = np.zeros((1,np.shape(x)[1]+1))
        # We find all end nodes and collect the indexes from 'subset' as well as
        # the vector of coefficients from 'beta'. Then, we create a copy of the
        # vector 'beta' for each of the indexes in 'subset'. Finally, we sort the
        # coefficients based on the indexes and store them in 'beta_hat' of the 
        # current tree.
        for i in range(0,self.depth):
            for j in range(0,len(self.leaves[i])):
                if self.leaves[i][j].endnode=='yes' or i==self.depth-1:
                    # We access the indexes from the end node.
                    x_sub = x[self.leaves[i][j].subset]
                    n = np.shape(x_sub)[0]
                    indeces = self.leaves[i][j].subset.copy()
                    # We create copies of the 'beta' vector for each entry in
                    # 'subset'. We then concatenate with the result of the 
                    # other end nodes.
                    betas = np.tile(self.leaves[i][j].beta.T,(n,1))
                    results_leaf = np.concatenate((np.expand_dims(indeces,axis=1),betas), axis=1)
                    beta_temp = np.concatenate((beta_temp,results_leaf))
                else:
                    continue
        # sort all rows by the indeces (first column) so that each row corresponds
        # to the data points of the initial feature set x.
        beta_temp = beta_temp[1:,:]
        beta_temp = beta_temp[beta_temp[:,0].argsort()]
        # drop the column of indeces
        self.beta_hat = beta_temp[:,1:]
        
    
    # Method to obtain the coefficients for new input data. It takes new data
    # 'x' as input and, for each data point, it follows the path from the starting
    # node to determine the end node and the corresponding vector of coefficients.
    def predict(self,x):
        pred = np.zeros((np.shape(x)[0],np.shape(x)[1]))
        # This 'for loop' selects each input data point at a time.
        for k in range(0,np.shape(x)[0]):
            i=0
            j=0
            end = 0
            xx = x[k,:]
            # This 'while loop' stops when we encounter an end node or when 
            # we reach the maximum depth.
            while end == 0:
                # If we reach an end node or maximum depth, we retrieve the 
                # coefficients and end the 'while loop'
                if self.leaves[i][j].endnode == 'yes' or i == self.depth-1:
                    pred[k,:] = np.squeeze(self.leaves[i][j].beta)
                    end = 1 
                # Otherwise, we climb down the tree by comparing the split variable
                # and split value of the current leaf in layer 'i'.
                else:
                    var = self.leaves[i][j].split_var
                    val = self.leaves[i][j].split_val
                    # If the condition is 'true' we are in a 'left' split. 
                    # We move to the lower layer by increasing 'i' by 1.
                    # Because we are in a 'left' split, the index of the next 
                    # node is given by 'id_next_leaf'.
                    if xx[var]<=val:
                        j = self.leaves[i][j].id_next_leaf
                        i = i+1
                    # If the condition is 'false' we are in a 'right' split. 
                    # We move to the lower layer by increasing 'i' by 1.
                    # Because we are in a 'right' split, the index of the next 
                    # node is given by 'id_next_leaf+1'.    
                    else:                    
                        j = self.leaves[i][j].id_next_leaf+1
                        i = i+1          
        return pred
     
                         
# Train and test the algorithm.
# We calculate the in-sample MSE, the out-of-sample MSE and the out-of-sample
# RMSPE for different combinations of the hyperparameters. Below, we vary 
# 2 hyperparameters. For the MSE, we store the value after each iteration 
# of the XGBOOST to examine how the values evolve.
mse1 = np.ones((16,100))
mse2 = np.ones((16,100))
rmspe1 = np.ones((4,4))
gamma = np.array([0.05,0.1,0.15,0.2])
llambda = np.array([0.3,0.6,0.9,1.2])
learning_rate = np.array([0.02,0.2])
depth = np.array([3,5])
for k in range(0,4):
    for j in range (0,4):
        # The matrix of coefficients is initialized with 'beta_0'
        # 'beta_hat' stores the matrix of coefficients for the testing procedure.
        # The final output of the XGBOOST is a sum over all trees. Hence, we 
        # calculate and store the coefficients relative to the test dataset 
        # during each iteration and sum them to the coefficients of the previous
        # iteration.
        beta_prev = beta_0 
        beta_hat = beta_0[0:len(x_test),:]
        # This 'for loop' creates a tree in each iteration.
        for i in range(0,100):
            tree1 = tree(5,beta_prev) # Select the tree depth here
            tree1.build_tree(x_train,y_train,llambda[j],gamma[k])  # Select lambda and gamma here
            # With 'estimate_betas' we obtain the coefficients relative to the train
            # dataset
            tree1.estimate_betas(x_train)
            # With 'predict' we obtain the coefficients relative to the test dataset
            beta_test = tree1.predict(x_test)
            # The coefficients relative to the training dataset are multiplied by 
            # the learning rate and summed to their previous value. 'beta_hat' will be 
            # used to calculate the in-sample MSE.
            beta_prev = beta_prev + 0.02*tree1.beta_hat
            # The coefficients relative to the test dataset are multiplied by 
            # the learning rate and summed to their previous value. 'beta_prev' will be 
            # used to calculate the out-of-sample MSE and RMSPE.
            beta_hat = beta_hat + 0.02*beta_test
            mse1[j+4*k,i]=np.mean(np.square(y_train - np.diagonal(x_train@beta_prev.T))) # in-sample MSE
            mse2[j+4*k,i]=np.mean(np.square(y_test - np.diagonal(x_test@beta_hat.T))) # out-of-sample MSE
        rmspe1[k,j] = rmspe(y_test,np.diagonal(x_test@beta_hat.T)) # out-of-sample RMSPE


# end_time = time.time() # The runtime is calculated up to this point
# elapsed_time = end_time - start_time # Calculates the total runtime



