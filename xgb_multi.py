# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 16:08:03 2023

@author: sam37
"""

# Multi-output variation of the AR-XGBOOST algorithm (MOAR-XGBOOST).
# 200 bins for the approximate split finding.
# Splits were executed 1 layer at a time.

import os
import pandas as pd
import numpy as np
import numpy.matlib
from dateutil.parser import parse
import matplotlib.pyplot as mp
import statsmodels.api as sm
import time
from sklearn.metrics import r2_score
from scipy.linalg import block_diag


# start_time = time.time()  # Used to measure the runtime starting from this point


path = 'C:\\Users\\sam37\\Desktop\\Master-Thesis\\Data' # Directory to access data
os.chdir(path)


# Dataset containing prices
data = pd.read_csv('3571from1995ns.csv') 
# Dataset containing fundamentals 
finstat = pd.read_csv('finstat.csv')  
finstat = finstat.rename(columns={'dvpspq':'div','epspxq':'eps','oancfy':'cf','oibdpq':'opin','saleq':'sales'})


# Delete the stocks for which there is no full time series and that contain
# too many nan or inf values. We only keep IBM, UIS, MU, AAPL, CGNX
cross_tickers = list(set(data['TICKER']) & set(finstat['tic']))
multi_tics = list(['IBM', 'UIS', 'MU', 'AAPL', 'CGNX'])
unique_tic = data['TICKER'].unique()
tic_to_drop = list(set(unique_tic)-set(multi_tics))
for tic in tic_to_drop:
    data = data.drop(data[data['TICKER']==tic].index)


# Because some time series are fragmented we need to regroup them
indexx = np.zeros(len(data))
i=0
data['indexx']=indexx
for tic in data['TICKER'].unique():
    #data[data['TICKER']==tic]['indexx']=list(range(i,len(data[data['TICKER']==tic])))
    data.loc[data['TICKER']==tic,'indexx']=list(range(i,i+len(data[data['TICKER']==tic])))
    i = i+len(data[data['TICKER']==tic])
data=data.sort_values(by=['indexx'])

# Book value (in millions)
finstat['bookv'] = finstat['atq'] - finstat['ltq']
# Market value
data['mktv'] = data['PRC']*data['SHROUT']


# Fundamentals are quarterly data and we need daily financial ratios. 
# Therefore we need to pair quarterly fundamentals to daily prices.
# This is done by quarte_to_day.
# 'Quarterly' is the dataset of fundamentals, 'daily' is the dataset of prices
# and 'measure' is the label of the fundamental we want to convert to daily series.
# It returns 'daily_fin', the series of daily fundamentals, and save_start, 
# the first date shared by both the daily and quarterly series.
def quarter_to_day(quarterly, daily, measure):
    date2 = parse(daily['date'].iloc[0]) # Initial date of daily series
    init_fin = parse(quarterly['datadate'].iloc[0]) # Initial date of quarterly series
    j=0
    k=0
    daily_fin = [] 
    # We move the date of daily series forward until it matches the date of the
    # quarterly series.
    while date2<init_fin:
        daily_fin.append(0)
        j = j+1
        if j>(len(daily)):
            break
        date2 = parse(daily['date'].iloc[j])
    save_start = daily['date'].iloc[j]
    # For each date of the quarterly series, we repeat the fundamentals on a daily
    # frequency until the date matches the date of new fundamentals.
    for d1 in quarterly['datadate']:
        k=k+1
        if k>(len(quarterly)-1) or j>(len(daily)-1):
            break
        while date2 < parse(quarterly['datadate'].iloc[k]):
            j=j+1
            daily_fin.append(quarterly[measure].iloc[k-1])             
            if j>(len(daily)-1):
                break          
            date2 = parse(daily['date'].iloc[j])
    return daily_fin, save_start


# We use 'quarter_to_day' to create the daily series for the 6 fundamentals below
# and store the date of the start of the sensible part of the time series.
# We do not consider time series before fundamentals are available.
date_start = []
for measure in ['bookv','div','eps','cf','opin','sales']:
    temp = []
    for tic in data['TICKER'].unique():
        a,b = quarter_to_day(finstat[finstat['tic']==tic],data[data['TICKER']==tic],measure)
        temp.extend(a)
        date_start.append(b)
    data[measure]=temp
    

# Out of all the starting dates, we select the latest ('start_date') to 
# synchronize the time series and to have matching dates.
start_date = '1900-01-01'
for date in date_start:
    if parse(start_date)<parse(date):
        start_date = date
    

# We drop all rows before 'start_date' for each stock.
ind = []
for tic in data['TICKER'].unique():
    datatemp = data.loc[data['TICKER']==tic]
    start_index = datatemp[datatemp['date']==start_date].index[0]
    ind.extend(datatemp.loc[:start_index-1,:].index)
data = data.drop(ind)


# Build financial ratios dataframe
fratios = pd.DataFrame()
fratios['ticker'] = data['TICKER']
fratios['bm'] = data['bookv']/data['mktv']
fratios['dy'] = data['div']/data['PRC']
fratios['pe'] = data['PRC']/data['eps']
fratios['poin'] = data['PRC']/data['opin']
fratios['pcf'] = data['PRC']/data['cf']
fratios['psales'] = data['PRC']/data['sales']


# 'lag_input' is a function that creates the inputs for the regressions, namely
# the matrix 'x' of regressors and the vector 'y', the dependent variable.
# 'y' contains returns, 'x' contains lagged returns and lagged financial ratios.
# 'data2' is the dataframe that contains prices under the label 'PRC'.
# 'lag' is the number of lagged days for the variables in 'x'.
# 'return_span' is the time horizon of returns.
def lag_input(data2,lag,return_span):
    returns = (np.array(data2['PRC'][return_span-1:])-np.array(data2['PRC'][:-(return_span-1)]))/np.array(data2['PRC'][:-(return_span-1)])
    y = returns[lag:]
    x = fratios.loc[fratios['ticker']==data2['TICKER'].iloc[0]]
    x = x.iloc[return_span-1:-lag,1:]
    x = x.to_numpy()
    lagged_y = np.array(returns[:-lag])
    lagged_y = np.expand_dims(lagged_y,axis=1)
    x = np.concatenate((lagged_y,x),axis=1)
    return y,x


# Function that forward fills nan values 
def fill_nan(x):
    row,col = x.shape
    mask = np.isnan(x)
    for i in range(1,row):
        for j in range(col):
            if mask[i][j]:
                x[i][j] =x[i-1][j]
    return x  


# Function that forward fills inf values 
def fill_inf(x):
    row,col = x.shape
    mask = np.isinf(x)
    for i in range(1,row):
        for j in range(col):
            if mask[i][j]:
                x[i][j] =x[i-1][j]
    return x  


# Function that calculates the RMSPE. 'y' is the real value, 'y_hat'
# is the estimate.
def rmspe(y,y_hat):
    zeroid = np.nonzero(y)[0]
    rmspe = np.sqrt(np.mean(np.square((y[zeroid]-y_hat[zeroid])/y[zeroid])))
    return rmspe

# This function creates a block matrix with a number of identical blocks equal 
# to the number of stocks analyzed. Each block contains 2 lags of the financial ratios
# and 2 lags of the returns averaged over all stocks. 'data' is the dataframe with 
# prices. 'end' and 'start' determine the rows to select for the training and test
# split. 'lag' is the number of days between the regressors and 
# the target variable. 'return_span' is the time horizon of returns.
def reg_input_average(data,start,end,lag,return_span):
    y=[]
    N = len(data['TICKER'].unique())
    tics = data['TICKER'].unique()
    y1,x1 = lag_input(data.loc[data['TICKER']==tics[0]],lag,return_span)
    y2,x2 = lag_input(data.loc[data['TICKER']==tics[0]],lag*2,return_span)    
    y2 = y2[start:end]
    y = np.concatenate((y,y2),axis=0)
    x1 = np.concatenate((x1[lag:],x2),axis=1) 
    x1 = fill_nan(x1)
    x1 = x1[start:end,:]
    for i in range(1,len(data['TICKER'].unique())):
        ytemp,xtemp = lag_input(data.loc[data['TICKER']==tics[i]],lag,return_span)
        y2,x2 = lag_input(data.loc[data['TICKER']==tics[i]],lag*2,return_span)    
        y2 = y2[start:end]
        y = np.concatenate((y,y2),axis=0)
        xtemp = np.concatenate((xtemp[lag:],x2),axis=1)
        xtemp = xtemp[start:end,:]
        x1 = x1 + xtemp
        i = i+1
    x1 = x1/N
    x = block_diag(*([x1]*N)) 
    return y,x
    
    
# We build the input matrices for training and testing.
# The matrix of regressors 'x' will contain 2 lags of the financial ratios
# and 2 lags of the returns. 'y' will be a vector of target returns. 
N = len(data['TICKER'].unique())  # Number of stocks
data_T = int(len(data)/N) # Length of single stock time series
# Returns over 5 days and regressors are 5 days behind the target variable.
y_train, x_train = reg_input_average(data,0,data_T-1000,5,5) 
y_test, x_test = reg_input_average(data,data_T-1000,data_T-500,5,5)


# This function calculates the realized volatility for 'log_returns' over
# the time horizon corrisponding to 'span'.  
def real_vol(log_returns,span):
    vol = np.zeros((len(log_returns)-span+1,1))
    for i in range(0,len(log_returns)-span+1):
        vol[i] = np.sqrt(np.sum(np.square(log_returns[i:i+span])))
    return vol


# This function creates a block matrix with a number of identical blocks equal 
# to the number of stocks analyzed. Each block contains 2 lags of the financial ratios
# and three realized volatility components averaged over all stocks. The realized 
# volatility components are the 5,10 and 20 days realized volatility. 'data' is the 
# dataframe with prices. 'end' and 'start' determine the rows to select for the 
# training and test split. 'lag' is the number of days between the regressors and 
# the target variable. 'n_fratios_lag' is the number of lags for the finratios.
# 'vol_span' is a list of spans over which realized volatilites are calculated.
def reg_input_average2(data,fratios,start,end,n_fratios_lag,vol_span,lag):
    N = len(data['TICKER'].unique())
    tics = data['TICKER'].unique()
    y=[]
    x=np.zeros(((end-start),(fratios.shape[1]-1)*n_fratios_lag+len(vol_span)))
    for i in range(0,len(data['TICKER'].unique())):
        subdata = data.loc[data['TICKER']==tics[i]]
        log_returns = np.log(np.array(subdata['PRC'][1:])/np.array(subdata['PRC'][0:-1]))
        ytemp = real_vol(log_returns,vol_span[0])[vol_span[-1]-vol_span[0]+lag:]
        y.append(ytemp[start:end])
        xtemp = []
        for j in range(0,n_fratios_lag):
            xtemp.append(np.array(fratios.loc[fratios['ticker']==tics[i]].iloc[vol_span[-1]-lag*j:-lag*(j+1),1:]))
        for k in range(0,len(vol_span)):
            xtemp.append(real_vol(log_returns,vol_span[k])[vol_span[-1]-vol_span[k]:-lag])
        xtemp = np.hstack(xtemp)
        x=x+xtemp[start:end]
    y = np.vstack(y)
    x = x/N
    xx = block_diag(*([x]*N))
    return y[:,0],xx


# **Uncomment this section to work with realized volatility**
# We separate the data into training and test sets.
N = len(data['TICKER'].unique())  # number of stocks
data_T = int(len(data)/N)
y_train, x_train = reg_input_average2(data,fratios,0,data_T-1000,2,[5,10,20],5)
y_test, x_test = reg_input_average2(data,fratios,data_T-1000,data_T-500,2,[5,10,20],5)


# Fill all nan and inf values in the data.
x_train = fill_nan(x_train)
x_train = fill_inf(x_train)
x_test = fill_nan(x_test)
x_test = fill_inf(x_test)
        

# Regress y on x to find a first estimate of beta, beta_0
n = len(x_train)
m = x_train.shape[1]
model = sm.OLS(y_train,x_train)
leaff = model.fit()
beta_0 = leaff.params
beta_0 = np.ones((n,m))*beta_0.T


# Functions that calculate the gradient and the hessian for each data point.
# Both gradients and hessians are calculated from the mean squared error.
def gradient(y,x,beta):
    yhat = np.zeros((len(y),1))
    for i in range(0,len(y)):
        yhat[i] = x[i,:]@beta[i,:].T
    grad = -(y-yhat[:,0])
    return np.expand_dims(grad,axis=1)
def hessian(x):
    hess = 1
    return hess


# To split a leaf we need to compare the possible scores, one for each possible split.
# In order to do so, we need to know what the subset of 'x' for the current leaf is. The argument
# 'x' in this function already contains the current subset.
# Then, calculate the score using the current subset, the subset resulting from split 1
# and the subset resulting from split 2. 'split_val' is the value of the split, 
# 'split_var' is the feature that determines the split, grad and hess are the gradients
# and hessians, 'y' is the vector of target variables corresponding to the subset 
# of the current leaf. 'x_tic' is the single block that forms the block matrix 'x'.
# 'n' is the number of stock. It returns the score and the two subsets resulting 
# from the split.
def score(split_val,split_var,grad,hess,llambda,gamma,x,y,x_tic,n):
    #select the two possible subsets and calculate the score reuslting form this split
    split2 = x_tic[:,split_var]>split_val
    split1 = x_tic[:,split_var]<=split_val
    split2 = np.squeeze(np.tile(split2,(1,n)))
    split1 = np.squeeze(np.tile(split1,(1,n)))
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
    def find_splitt(self,llambda,gamma,x,y,beta,n,m):
        gainn = 0
        # 'sub1' and 'sub2' store the resulting subsets of the split.
        sub1 = []
        sub2 = []
        T = int(len(x)/n)
        # If 'x' has more than 200 data points we proceed with the approximate split
        # finding algorithm. Otherwise, we employ the exact greedy split finding algorithm.
        x_tic = x[:T,:m]
        if len(x)>200:
            # Calculate the gradient, the 'xg' and 'xhx' terms for the dataset
            # of the parent node.
            grad = gradient(y,x,beta)
            gx = grad.T@x
            xhx = x.T@x
            # Calculate the loss corresponding to the parent node.
            L = -1/2*gx@np.linalg.pinv(xhx+llambda*np.eye(np.shape(x)[1]))@gx.T
            # In this 'for loop' we caclulate the 'gain' for the split corresponding
            # to each of the 200 bins for each feature.
            for i in range(0,x_tic.shape[1]): # This loop runs through the features
                # Create the 200 percentiles for feature 'i'
                steps = np.linspace(100/200,100,num=200)
                percs = np.percentile(x_tic[:,i],steps)
                # Create the bins for the 'xg' and 'xhx' terms and a bin,
                # 'bins_id', that stores the index of the corresponding data points               
                gx_bin = []
                xhx_bin = []
                bins_id = []
                # Because the first bin corresponds to a left-sided region and not 
                # an interval, this step takes place before the next 'for loop'.
                # We first find the indexes corresponding the left-sided region and 
                # store the 'xg' and 'xhx' terms corresponding to those indexes.
                id_temp = np.where(x_tic[:,i]<=percs[0])[0]
                bins_id.append(np.concatenate([id_temp+T*i for i in range(0,n)],axis=0))
                gx_bin.append(grad[bins_id[0]].T@x[bins_id[0]])
                xhx_bin.append(x[bins_id[0],:].T@x[bins_id[0],:])
                # This loop runs through the remaining percentiles to fill the bins
                for k in range(1,200):
                    id_temp = np.where(np.logical_and(x_tic[:,i]>percs[k-1], x_tic[:,i]<=percs[k]))[0]
                    bins_id.append(np.concatenate([id_temp+T*i for i in range(0,n)],axis=0))
                    gx_bin.append(grad[bins_id[k]].T@x[bins_id[k]])
                    xhx_bin.append(x[bins_id[k],:].T@x[bins_id[k],:])
                # We turn the lists into numpy arrays
                gx_bin = np.array(gx_bin)
                xhx_bin = np.array(xhx_bin)
                # We initialize the 'xg' and 'xhx' terms for the left region of the split
                leftgx = 0
                leftxhx = 0
                # In this 'for loop' we add the 'xg' and 'xhx' terms of bin 'j' 
                # to the previous totals corresponding to the left region. In other words,
                # in each iteration we calculate the cumulative 'xg' and 'xhx' of the left
                # and right regions for a certain split.
                for j in range(0,200-1):
                    # The new 'xg' and 'xhx' terms for the left region are equal to 
                    # the previous left-totals plus the values in bin 'j'
                    leftgx = leftgx + gx_bin[j]
                    leftxhx = leftxhx + xhx_bin[j,:,:]
                    # The new 'xg' and 'xhx' terms for the right region are equal to 
                    # the overall totals minus the current left totals 
                    rgx = gx - leftgx
                    rxhx = xhx - leftxhx
                    # Calculate the losses corresponding to the current left 
                    # and right regions. Then calculate the gain defined by 'scoree'
                    Ll = -1/2*leftgx@np.linalg.pinv(leftxhx+llambda*np.eye(np.shape(x)[1]))@leftgx.T
                    Lr = -1/2*rgx@np.linalg.pinv(rxhx+llambda*np.eye(np.shape(x)[1]))@rgx.T             
                    sscore = L - Ll - Lr - gamma
                    # If the current gain is larger than the previous largest gain,
                    # we store the indexes corresponding to the current two regions into
                    # 'sub1' and 'sub2'. We also store the feature and the value of
                    # the split into 'split_var_temp' and 'split_val_temp' respectively.
                    if sscore>gainn:
                        left_x = np.squeeze(np.hstack(bins_id[:j+1]))
                        right_x = np.squeeze(np.hstack(bins_id[j+1:]))
                        gainn = sscore
                        split_val_temp = percs[j]
                        split_var_temp = i
                        sub1 = left_x
                        sub2 = right_x
        else:
            # In this 'else' section is the exact greedy algorithm.
            # For each feature we first sort the values.
            for i in range(0,x_tic.shape[1]):
                sorted_x = np.sort(x_tic[:,i])
                # For each of the sorted values we calculate the average between the next 
                # value and itself. The average is then used to determine the split and
                # calculate the gain.
                for j in range(0,x_tic.shape[0]-1):
                    # grad and hess are inputs for gradients and hessians.
                    # Inputs are passed for the entire subset of the current leaf.
                    # Then, inside 'score', smaller subsets will be assigned to the 
                    # corresponding splits.
                    grad = gradient(y,x,beta)
                    hess = hessian(x)
                    value = (sorted_x[j+1]+sorted_x[j])/2
                    scoree, sub1_temp, sub2_temp = score(value,i,grad,hess,llambda,gamma,x,y,x_tic,n)
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
        
    
    # Function that calculates the beta coefficients for the current leaf. 'x_tot'
    # is the entire matrix of regressors. 'y_tot' is the entire vector of target 
    # variables. 'prev_beta_tot' is the entire matrix of beta coefficients from 
    # the previous tree. 'n_stocks' is the number of stocks. 'm' is the number 
    # of features.
    def calculate_beta(self,llambda,x_tot,y_tot,prev_beta_tot,n_stocks,m):
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
    
    
    def __init__(self,depth,prev_beta,n_stocks,m):
        self.depth = depth # Number of layers
        self.leaves = [[leaf([])]] # A list of 'leaf' objects that form the tree.
        self.prev_beta = prev_beta # Beta coefficients of the previous tree
        self.beta_hat = [] # Beta coefficients of the current tree
        self.n_stocks = n_stocks # Number of stocks
        self.m = m  # Number of features
    
    
    # Method that builds the tree. It takes the block matrix of regressors 'x',
    # the vector of target variables 'y' and the hyperparameters lambda
    # and gamma as arguments.
    def build_tree(self,x,y,llambda,gamma):
        # We create the initial node with the indexes of the full dataset
        # in 'subset' and calculate the set of coefficients corresponding
        # to the full dataset using the coefficients of the last tree.
        self.leaves[0][0].subset = np.arange(0,len(x))
        self.leaves[0][0].calculate_beta(llambda,x,y,self.prev_beta,self.n_stocks,self.m)
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
                if len(sub)<7*self.n_stocks:
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
                splitt1,splitt2, sub1, sub2 = self.leaves[i][j].find_splitt(llambda,gamma,sub_x,y_sub,beta_prev_sub,self.n_stocks,self.m) 
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
                self.leaves[i+1][-1].calculate_beta(llambda,x,y,self.prev_beta,self.n_stocks,self.m)
                self.leaves[i+1][-2].calculate_beta(llambda,x,y,self.prev_beta,self.n_stocks,self.m)
                self.leaves[i][j].id_next_leaf = len(self.leaves[i+1])-2  
                
                
    
    # Method that creates the 'beta_hat' matrix of the current tree.
    # Each row of the matrix is a set of coefficients for the corresponding
    # row of the input dataset 'x'. It is used to create a matrix of coefficients
    # where each row corresponds to a row in the training dataset 'x_train'.
    def estimate_betas(self,x):
        # We find all end nodes and collect the indexes from 'subset' as well as
        # the vector of coefficients from 'beta'. Then, we create a copy of the
        # vector 'beta' for each of the indexes in 'subset'. Finally, we sort the
        # coefficients based on the indexes and store them in 'beta_hat' of the 
        # current tree.
        beta_temp = np.zeros((1,np.shape(x)[1]+1))
        for i in range(0,self.depth):
            for j in range(0,len(self.leaves[i])):
                # stores the indeces of the subset for each leaf.
                # We only need the subset for the leaves that yield the beta_hat
                # we are going to use.
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
        T = int(len(x)/self.n_stocks)
        pred = np.zeros((np.shape(x)[0],np.shape(x)[1]))
        # x_tic collects a signle block of the block matrix
        x_tic = x[:T,:self.m]
        # This 'for loop' selects each input data point at a time.
        for k in range(0,np.shape(x_tic)[0]):
            i=0
            j=0
            end = 0  
            xx = x_tic[k,:]
            # This 'while loop' stops when we encounter an end node or when 
            # we reach the maximum depth.
            while end == 0:
                # If we reach an end node or maximum depth, we retrieve the 
                # coefficients and end the 'while loop'
                if self.leaves[i][j].endnode == 'yes' or i == self.depth-1:
                    for h in range(0,self.n_stocks):
                        pred[k+(T*h),:] = np.squeeze(self.leaves[i][j].beta)
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
# 2 hyperparameters. We store the values after each iteration 
# of the XGBOOST to examine how the values evolve.
mse1 = np.ones((4,30))
mse2= np.ones((4,30))
rmspe1 = np.ones((4,30))
M = int(m/N)
gammaa = np.array([0.1,0.3])
llambda = np.array([0.3,0.9])
depth = np.array([5,7])
learning_rate = np.array([0.02,0.08])
for k in range(0,2):
    for j in range (0,2):
        # The matrix of coefficients is initialized with 'beta_0'
        # 'beta_hat' stores the matrix of coefficients for the testing procedure.
        # The final output of the XGBOOST is a sum over all trees. Hence, we 
        # calculate and store the coefficients relative to the test dataset 
        # during each iteration and sum them to the coefficients of the previous
        # iteration.
        beta_prev = beta_0 
        beta_hat = beta_0[0:len(x_test),:]
        # This 'for loop' creates a tree in each iteration.
        for i in range(0,5):
            tree1 = tree(3,beta_prev,N,M) # Select the tree depth here
            tree1.build_tree(x_train,y_train,llambda[j],gammaa[k]) # Select lambda and gamma here
            # With 'estimate_betas' we obtain the coefficients relative to the train
            # dataset
            tree1.estimate_betas(x_train)
            # With 'predict' we obtain the coefficients relative to the test dataset
            beta_test = tree1.predict(x_test)
            # The coefficients relative to the training dataset are multiplied by 
            # the learning rate and summed to their previous value. 'beta_hat' will be 
            # used to calculate the in-sample MSE.
            beta_hat = beta_hat + 0.02*beta_test
            beta_prev = beta_prev + 0.02*tree1.beta_hat
            mse1[j+2*k,i]=np.mean(np.square(y_train - np.diagonal(x_train@beta_prev.T)))
            mse2[j+2*k,i]=np.mean(np.square(y_test - np.diagonal(x_test@beta_hat.T)))
            rmspe1[j+2*k,i] = rmspe(y_test,np.diagonal(x_test@beta_hat.T))


# end_time = time.time() # The runtime is calculated up to this point
# elapsed_time = end_time - start_time # Calculates the total runtime