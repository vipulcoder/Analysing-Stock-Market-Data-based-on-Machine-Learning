#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import datetime as dt


# In[6]:


#import the dataset
dp=pd.read_csv("C:/Users/vashisth/Downloads/GOOG.csv")


# In[7]:


dp


# In[8]:


dp[dp['high'] > 1500]


# In[9]:


dp[dp['volume'] > 4000000]


# In[10]:


sns.scatterplot(x='open',y='close',data=dp)
plt.show()


# In[11]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x='high',y='low',data=dp)
plt.show()


# In[12]:


sns.scatterplot(x='volume',y='date',data=dp)
plt.show()


# In[13]:


sns.set(rc={'figure.figsize':(15,10)})
sns.barplot(x='volume',y='adjVolume',data=dp)
plt.show()


# In[14]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x='date',y='high',data=dp)
plt.show()


# In[15]:


sns.set(rc={'figure.figsize':(15,10)})
sns.lineplot(x='date',y='volume',data=dp)
plt.show()


# In[16]:


y=dp[['open']]
x=dp[['high']]


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[19]:


x_train.head()


# In[20]:


x_test.head()


# In[21]:


y_train.head()


# In[22]:


y_test.head()


# In[23]:


#linear regression model
from sklearn.linear_model import LinearRegression


# In[24]:


lr=LinearRegression()


# In[25]:


lr.fit(x_train,y_train)


# In[26]:


y_pred=lr.predict(x_test)


# In[27]:


y_test.head()


# In[28]:


y_pred[0:10]


# In[29]:


from sklearn.metrics import mean_squared_error


# In[30]:


mean_squared_error(y_test,y_pred)


# In[31]:


#multiple linear regression
y=dp[['high']]
x=dp[['close','volume']]


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


# In[33]:


lr2=LinearRegression()


# In[34]:


lr2.fit(x_train,y_train)


# In[35]:


mean_squared_error(y_test,y_pred)


# In[ ]:




