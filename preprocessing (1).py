#!/usr/bin/env python
# coding: utf-8

# In[171]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization


# In[196]:


def preprocessing():
    '''missing values,checking data types'''
    df1=pd.read_csv('C:/Users/M V N POOJITHA/Downloads/SBAcase.csv') #loading dataset
    print("numbers of rows and columns:",df1.shape)
    print(df1.index)
    print(df1.info())
    print(df1.describe())
    x=df1.isnull().sum()
    return x
def catergorical():
    '''see the categorical data'''
    X=df1.iloc[:,:-1].values
    return X


# In[197]:


preprocessing()


# In[195]:


catergorical()


# In[ ]:




