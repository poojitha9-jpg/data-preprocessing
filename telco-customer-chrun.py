#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import library
import pandas  as pd #Data manipulation
import numpy as np #Data manipulation
import matplotlib.pyplot as plt # Visualization
import seaborn as sns #Visualization


# In[2]:


customer = pd.read_csv('C:/Users/M V N POOJITHA/Downloads/Telco-Customer-Churn.csv') #loading dataset
customer.head() #display dataset


# In[3]:


customer.shape


# In[4]:


customer.info()  #information of the dataset


# In[5]:


customer.index


# In[6]:


customer.columns


# In[7]:


print ("Rows     : " ,customer.shape[0])   #print no.of rows
print ("Columns  : " ,customer.shape[1])   #print no.of cols
print ("\nFeatures : \n" ,customer.columns.tolist())   #print columns list
print ("\nMissing values :  ", customer.isnull().sum().values.sum())    #print missing values
print ("\nUnique values :  \n",customer.nunique())   #print unique values


# In[8]:


customer.describe()


# In[9]:


#checking for missing values
plt.figure(figsize=(14,4))
sns.heatmap(customer.isnull(),cbar=False,cmap='viridis',yticklabels=False)
plt.title('Missing value in the dataset');


# In[10]:


customer['Churn'][:5] #print churn values upto index 5


# In[11]:


customer['Churn'] = customer['Churn'].replace({"Yes":1,"No":0})  #replacing '1' value for 'yes' and '0' value for 'no'
customer['Churn'][:5]


# In[12]:


cols = ['OnlineBackup', 'StreamingMovies','DeviceProtection','TechSupport','OnlineSecurity','StreamingTV']
for values in cols:
    customer[values] = customer[values].replace({'No internet service':'No'})


# In[13]:


customer['TotalCharges'] = customer['TotalCharges'].replace(" ",np.nan)

# Drop null values of 'Total Charges' feature
customer= customer[customer["TotalCharges"].notnull()]
customer= customer.reset_index()[customer.columns]

customer['TotalCharges'] = customer['TotalCharges'].astype(float)
customer['TotalCharges'].dtype


# In[14]:


customer['Churn'].value_counts().unique()  #print yes  and no values for chunk in an array


# In[15]:


customer.isnull().sum()


# In[16]:


X=customer.iloc[:,:-1].values
X


# In[17]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
X[:,0]=label_encoder.fit_transform(X[:,0])
X


# In[18]:


churn_data= pd.get_dummies(customer, columns = ['Contract','Dependents','DeviceProtection','gender',
                                                        'InternetService','MultipleLines','OnlineBackup',
                                                        'OnlineSecurity','PaperlessBilling','Partner',
                                                        'PaymentMethod','PhoneService','SeniorCitizen',
                                                        'StreamingMovies','StreamingTV','TechSupport'],
                              drop_first=True)
churn_data.head()      


# In[19]:


customer=pd.concat([customer,churn_data],axis=1)
customer


# In[20]:


from sklearn.preprocessing import StandardScaler

#Perform Feature Scaling on 'tenure', 'MonthlyCharges', 'TotalCharges' in order to bring them on same scale
standard = StandardScaler()
columns_for_ft_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']

#Apply the feature scaling operation on dataset using fit_transform() method
churn_data[columns_for_ft_scaling] = standard.fit_transform(churn_data[columns_for_ft_scaling])
churn_data.head()


# In[21]:


list(churn_data.columns)


# In[22]:


from sklearn.model_selection import train_test_split
# Split Into Training and Testing Sets
X = churn_data.drop(['Churn','customerID'], axis=1)
y = churn_data['Churn']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

