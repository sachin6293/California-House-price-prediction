#!/usr/bin/env python
# coding: utf-8

# In[80]:


# import libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[81]:


# load dataset 
data = pd.read_csv("E:\\python  learning\\udemy\\housing price\\housing.csv")


# In[82]:


data.head()


# In[83]:


data.info()


# In[84]:


data.shape


# In[85]:


data.isnull().sum()


# In[86]:


data["ocean_proximity"].unique()


# In[87]:


data.describe()


# In[88]:


data.hist(bins = 50,figsize = (20,10))
plt.show()


# In[89]:


median = data["total_bedrooms"].median()


# In[90]:


def impute_nan(data,variable,median):
    data[variable+"_median"]=data[variable].fillna(median)


# In[91]:


impute_nan(data,'total_bedrooms',median)
data.head()


# In[92]:


fig = plt.figure()
ax = fig.add_subplot(111)
data['total_bedrooms'].plot(kind='kde', ax=ax)
data.total_bedrooms_median.plot(kind='kde', ax=ax, color='red')
lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels, loc='best')


# In[93]:


print(data['total_bedrooms'].std())
print(data['total_bedrooms_median'].std())


# In[94]:


data.isnull().sum()


# In[95]:


data.columns


# In[96]:


data.drop(["total_bedrooms"], axis = 1, inplace = True)


# In[97]:


data.head()


# In[98]:


data.columns


# In[99]:


data.isnull().sum()


# In[100]:


data.corr()


# In[111]:


plt.figure(figsize = (9,9))
sns.heatmap(data.corr(),annot = True, cmap = "coolwarm")


# In[120]:


dummies = pd.get_dummies(data.ocean_proximity, drop_first = True)


# In[121]:


Mdata = pd.concat([data, dummies], axis = 1)


# In[123]:


Mdata


# In[124]:


final_data = Mdata.drop(["ocean_proximity"], axis = 1)


# In[125]:


final_data


# In[127]:


final_data.columns


# In[133]:


x = final_data.iloc[:,:-6].values
y = final_data.iloc[:,-6].values


# In[134]:


y


# In[135]:


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)


# In[136]:


# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[138]:


# Predicting the Test set results
y_pred = regressor.predict(x_test)


# In[139]:


# limit the predicted values to the 2 decimal points.
np.set_printoptions(precision = 2)


# In[140]:


# concate the real values and the predicted values.
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[142]:


#  find the accuracy of the model.
accuracy = regressor.score(x_test,y_test)
print(accuracy*100,'%')


# In[143]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




