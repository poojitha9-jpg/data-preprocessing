#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization


# In[2]:


df= pd.read_csv('C:/Users/M V N POOJITHA/Downloads/telecom-churn-bigml.csv') #loading dataset
df.head(2) #display dataset


# In[3]:


df.shape


# In[4]:


df.info() #datset information


# In[5]:


df.index


# In[6]:


df.columns


# In[7]:


print ("Rows     : " ,df.shape[0])   #print no.of rows
print ("Columns  : " ,df.shape[1])   #print no.of cols
print ("\nFeatures : \n" ,df.columns.tolist())   #print columns list
print ("\nMissing values :  ", df.isnull().sum().values.sum())    #print missing values
print ("\nUnique values :  \n",df.nunique())   #print unique values


# In[8]:


df.describe()


# In[9]:


#checking for missing values
plt.figure(figsize=(14,4))
sns.heatmap(df.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[10]:


df.isnull().sum()


# In[11]:


#Separating churn and non churn customers
churn     = df[df["churn"] == bool(True)]
not_churn = df[df["churn"] == bool(False)]
#Dropping Account Length as it doesnt make a sense here
df = df.drop('account length',axis=1)
#Area Code
df['area code'].unique()


# In[12]:


#Replacing Yes/No values with 1 and 0
df['international plan'] = df['international plan'].replace({"yes":1,"no":0}).astype(int)
df['voice mail plan'] = df['voice mail plan'].replace({"yes":1,"no":0}).astype(int)


# In[13]:


#Voice-Mail Feautre Messages
print('unique vmail messages',df['number vmail messages'].unique())


# In[14]:


X=df.iloc[:,:-1].values
X


# In[15]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
X[:,0]=label_encoder.fit_transform(X[:,0])
X


# In[16]:


dummy=pd.get_dummies(df['churn'])
dummy


# In[17]:


df=pd.concat([df,dummy],axis=1)
df


# In[18]:


from sklearn.model_selection import train_test_split
# Split Into Training and Testing Sets

# Separate out the features and targets
features = df.drop(columns='churn')
targets = pd.DataFrame(df['churn'])

# Split into 80% training and 20% testing set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size = 0.2, random_state = 42)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

