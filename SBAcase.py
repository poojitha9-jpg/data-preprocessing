#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization


# In[2]:


df1= pd.read_csv('C:/Users/M V N POOJITHA/Downloads/SBAcase.csv') #loading dataset
df1.head(2) #display dataset


# In[3]:


df1.shape


# In[4]:


df1.info()


# In[5]:


df1.index


# In[6]:


df1.columns


# In[7]:


print ("Rows     : " ,df1.shape[0])   #print no.of rows
print ("Columns  : " ,df1.shape[1])   #print no.of cols
print ("\nFeatures : \n" ,df1.columns.tolist())   #print columns list
print ("\nMissing values :  ", df1.isnull().sum().values.sum())    #print missing values
print ("\nUnique values :  \n",df1.nunique())   #print unique values


# In[8]:


df1.describe()


# In[9]:


#checking for missing values
plt.figure(figsize=(14,4))
sns.heatmap(df1.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[10]:


df1.isnull().sum()


# In[11]:


df1.dropna(inplace=True)
df1.isnull().sum()


# In[12]:


df1.shape


# In[13]:


X=df1.iloc[:,:-1].values
X


# In[14]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
X[:,0]=label_encoder.fit_transform(X[:,0])
X


# In[15]:


dummy=pd.get_dummies(df1['MIS_Status'])
dummy


# In[16]:


df1=pd.concat([df1,dummy],axis=1)
df1


# In[17]:


from sklearn.model_selection import train_test_split
# Split Into Training and Testing Sets

# Separate out the features and targets
features = df1.drop(columns='MIS_Status')
targets = pd.DataFrame(df1['MIS_Status'])

# Split into 80% training and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

