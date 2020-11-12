#!/usr/bin/env python
# coding: utf-8

# In[146]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization

data=pd.read_csv('C:/Users/M V N POOJITHA/Downloads/credit_train.csv') #loading dataset


# In[147]:


def preprocess(data):
    '''checking missing values'''
    cnt=data.isnull().sum()
    percent=cnt/len(data)*100
    table=pd.concat([pd.DataFrame(cnt),pd.DataFrame(percent)],axis=1)
    table.columns=['counts','percentage']
    table.sort_values('counts',ascending=False,inplace=True)
    return table
def threshold(data):
    '''remove missing values'''
    table=preprocess(data)
    table=table[table['percentage']>=25]
    data.dropna(inplace=True)
    return data
def categorical(data):
    '''see the categorical data'''
    data=data.iloc[:,:-1].values
    return data


# In[148]:


preprocess(data)


# In[151]:


threshold(data)


# In[150]:


categorical(data)


# In[ ]:




