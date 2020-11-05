#!/usr/bin/env python
# coding: utf-8

# In[63]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization
data=pd.read_csv('C:/Users/M V N POOJITHA/Downloads/credit_train.csv') #loading dataset


# In[73]:


def preprocessing():
    '''missing values,checking data types'''
    
    print("numbers of rows and columns:",data.shape)
    print(data.index)
    print(data.info())
    print(data.describe())
    print(data.dtypes)
    print(type(data))
    
    #df1.dropna(inplace=True)
    #z=df1.isnull().sum()
    new_data = data.dropna(axis = 0, how ='any') 
  
    # comparing sizes of data frames 
    print("Old data frame length:", len(data), "\nNew data frame length:",  
           len(new_data), "\nNumber of rows with at least 1 NA value: ", 
           (len(data)-len(new_data))) 
    print("missing values:",data.isnull().sum())
    print("remove missing values:",new_data.isnull().sum())       
    
def catergorical():
    '''see the categorical data'''
    X=data.iloc[:,:-1].values
    return X


# In[74]:


preprocessing()


# In[66]:


catergorical()

