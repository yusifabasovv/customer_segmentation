#!/usr/bin/env python
# coding: utf-8

# You have to split customer ID's to 4 segment:
# 1. Loyal customers
# 2. New customers
# 3. You can't lose them
# 4. Lost costumers

#  

# Importing Libraries

# In[2]:


import pandas as pd
import numpy as np
import random 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[305]:


#Reading csv file
df=pd.read_excel("original_data _sifarisler.xlsx")
df.head()


# ### Data Wrangling

# In[297]:


df.shape


# In[300]:


#Checking Missing Values
df.isnull().sum()


# In[299]:


df.dtypes


# As we can see sifarisci column is object, we need it as int , so let's reformat it

# In[306]:


#We will use to_numeric technic as there is some string values in data, which we will convert them to nan
df["sifarisci"]=pd.to_numeric(df["sifarisci"],errors='coerce')

print(df.isnull().sum())
print(df.dtypes)


# So, we have sifarisci as float and our missing values increased due to strings in column

# In[307]:


#Cheking duplicates
df.duplicated().sum()


# In[308]:


# Lets drop them 
df.drop_duplicates(inplace=True)
df.duplicated().sum()


# In[309]:


print(df["sifarisci"].min())
df["sifarisci"].max()


# Machine Learning models especially K-means clustering can not handle this types of data column as we see that there is huge difference between values, Another point and most important point is that sifarisci column is customer's IDs , which is identical to each customers

# In[310]:


#Check duplicates in sifarisci column
df["sifarisci"].duplicated().sum()


# So we have 447 duplicated customers out of 1999 rows. We can encode them as 1, in others words as loyal customers , and others as 0, which so called not loyal customers. Then we can easily use ID's column in our model

# In[311]:


#Encode loyal customers as 1
for i in list(df["sifarisci"][df["sifarisci"].duplicated()].values):
    df["sifarisci"].replace(i,1,inplace=True)
    


# In[312]:


#And others as 0
for i in list(df["sifarisci"].values):
    if i>1:
        df["sifarisci"].replace(i,0,inplace=True)
        


# In[313]:


df["sifarisci"].unique()


# So now we have 2 unique sifarisci values

# In[314]:


df["sifarisci"].value_counts()


# In[334]:


df["sifarisci"].hist(bins=2)


# In[315]:


#Let's replace missing Satis values with their mean
df["Satis (azn)"].replace(np.nan,df["Satis (azn)"].mean(),inplace=True)
print(df.isnull().sum())


# In[341]:


df["Satis (azn)"].hist()


# Tarix column is in datetime format which our model can not handle it, so we need to change it.The best way can be seperate each date as year, month and day

# In[316]:


df["tarix_year"]=df["tarix"].dt.year
df["tarix_month"]=df["tarix"].dt.month
df["tarix_day"]=df["tarix"].dt.day


# In[317]:


df.drop("tarix",axis=1,inplace=True)
df


# In[318]:


#We can encode year as 1 or 0
df = pd.get_dummies(df, columns=['tarix_year'], drop_first=True, prefix='year')
df


# In[333]:


plt.hist(df["tarix_month"],bins=12)


# In[330]:


plt.hist(df["year_2020"],bins=2)


# We can easily see that most of sales reached to peak in between 6th and 8th months of year 2019

# ### Feature scaling and Model Building 
# For better model performance,Some Clustering models as well as K-Means requires data in a standard form which means we will give 0 mean and 1 variance to data

# In[342]:


X = df.values[:]
a = StandardScaler().fit_transform(X)
a


# In[343]:


#K-means model
clusterNum = 4
k_means = KMeans( n_clusters = clusterNum)
k_means.fit(a)
labels = k_means.labels_
print(labels)


# In[344]:


df["Clusters"]=labels


# In[346]:


df["Clusters"].value_counts()


# In[359]:


df["Clusters"].hist()


# In[360]:


df.loc[df["Clusters"] == 0]


# In[361]:


df.loc[df["Clusters"] == 1]


# In[362]:


df.loc[df["Clusters"] == 2]


# In[363]:


df.loc[df["Clusters"] == 3]


# In[364]:


df.groupby("Clusters").mean()


# # We can see from table :
# ###    - Sifarisci column has 1 mean in cluster 2, which means that all values in that cluster is loyal customers who has regular relations with company
# ###    - Cluster 3 has the higest mean values for sales, so we can define it as customers who we should not lose them as they contribute more to our company
# ###    - We can easily define cluster 1 as new customers, all of them is in 2020.
# ###    - Cluster 0 has the lowest date month value which means that customers in that cluster are not likely in relation with company , so we can define them as lost customers 

# In[358]:


#Visualization of model
from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3],X[:2] ,c= labels.astype(np.float))


# In[ ]:




