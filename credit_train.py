#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization


# In[2]:


credit = pd.read_csv('C:/Users/M V N POOJITHA/Downloads/credit_train.csv') #loading dataset
credit.head(2) #display dataset


# In[3]:


credit.shape


# In[4]:


credit.info()  #information of dataset


# In[5]:


credit.index


# In[6]:


credit.columns


# In[7]:


print ("Rows     : " ,credit.shape[0]) #print no.of rows
print ("Columns  : " ,credit.shape[1]) #print no.of cols
print ("\nFeatures : \n" ,credit.columns.tolist()) #print columns list
print ("\nMissing values :  ", credit.isnull().sum().values.sum())  #print missing values
print ("\nUnique values :  \n",credit.nunique()) #print unique values


# In[8]:


credit.describe() 


# In[9]:


#checking for missing values
plt.figure(figsize=(14,4))
sns.heatmap(credit.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[10]:


credit.isnull().sum()


# In[11]:


credit.dropna(inplace=True)
credit.isnull().sum()


# In[12]:


credit.shape


# In[13]:


# Correlations between Features and Target

correlations_data = credit.corr()['Credit Score'].sort_values(ascending=False)   # Find all correlations and sort 
print(correlations_data.tail)    # Print the correlations


# In[14]:


X=credit.iloc[:,:-1].values


# In[15]:


X


# In[16]:


from sklearn.preprocessing import LabelEncoder


# In[17]:


label_encoder=LabelEncoder()


# In[18]:


X[:,0]=label_encoder.fit_transform(X[:,0])


# In[19]:


X


# In[20]:


dummy=pd.get_dummies(credit['Loan Status'])


# In[21]:


dummy


# In[22]:


credit=pd.concat([credit,dummy],axis=1)


# In[23]:


credit


# In[25]:


# # # Split Into Training and Testing Sets
from sklearn.model_selection import train_test_split
# Separate out the features and targets
features = credit.drop(columns='Loan Status')
targets = pd.DataFrame(credit['Loan Status'])

# Split into 80% training and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[ ]:




