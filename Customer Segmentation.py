#!/usr/bin/env python
# coding: utf-8

# # Mall Customer Segmetation

# In[2]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv(r"C:\Users\jki\Downloads\customer segemtation\Mall-Customer-Segmentation\Mall_Customers.csv")
df.head()


# # Univariate Analysis
# 

# In[3]:


df.describe()


# In[4]:


sns.distplot(df['Annual Income (k$)']);


# In[5]:


df.columns


# In[6]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.distplot(df[i])


# In[9]:


import seaborn as sns

# Make sure df['Gender'] is a categorical variable
df['Gender'] = df['Gender'].astype('category')

sns.kdeplot(data=df, x='Annual Income (k$)', shade=True, hue='Gender')


# In[52]:


import seaborn as sns
import matplotlib.pyplot as plt

columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for i in columns:
    plt.figure()
    for gender in df['Gender'].unique():
        sns.kdeplot(df[df['Gender'] == gender][i], shade=True, label=gender)
    plt.title(f'Kernel Density Estimation for {i}')
    plt.legend()


# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# Create a single figure with multiple subplots
fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(15, 5))

# Loop through each column and plot the KDE
for i, col in enumerate(columns):
    sns.kdeplot(data=df, x=col, shade=True, hue='Gender', ax=axes[i])
    axes[i].set_title(col)  # Set title for each subplot

plt.tight_layout()
plt.show()


# In[12]:


columns = ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[13]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis
# 

# In[14]:


sns.scatterplot(data=df, x='Annual Income (k$)',y='Spending Score (1-100)' )


# In[15]:


#df=df.drop('CustomerID',axis=1)
sns.pairplot(df,hue='Gender')


# In[16]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[17]:


df.corr()


# In[18]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# # Clustering - Univariate, Bivariate, Multivariate
# 

# In[19]:


clustering1 = KMeans(n_clusters=3)


# In[20]:


clustering1.fit(df[['Annual Income (k$)']])


# In[21]:


KMeans(n_clusters=3)


# In[22]:


clustering1.labels_


# In[23]:


df['Income Cluster'] = clustering1.labels_
df.head()


# In[24]:


df['Income Cluster'].value_counts()


# In[25]:


clustering1.inertia_


# In[26]:


23517.33093093092


# In[27]:


intertia_scores=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    intertia_scores.append(kmeans.inertia_)


# In[28]:


intertia_scores


# In[29]:


plt.plot(range(1,11),intertia_scores)


# In[30]:


df.columns


# In[31]:


df.groupby('Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[32]:


#Bivariate Clustering


# In[33]:


clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
df['Spending and Income Cluster'] =clustering2.labels_
df.head()


# In[34]:


intertia_scores2=[]
for i in range(1,11):
    kmeans2=KMeans(n_clusters=i)
    kmeans2.fit(df[['Annual Income (k$)','Spending Score (1-100)']])
    intertia_scores2.append(kmeans2.inertia_)
plt.plot(range(1,11),intertia_scores2)


# In[36]:


centers =pd.DataFrame(clustering2.cluster_centers_)
centers.columns = ['x','y']


# In[37]:


plt.figure(figsize=(10,8))
plt.scatter(x=centers['x'],y=centers['y'],s=100,c='black',marker='*')
sns.scatterplot(data=df, x ='Annual Income (k$)',y='Spending Score (1-100)',hue='Spending and Income Cluster',palette='tab10')
plt.savefig('clustering_bivaraiate.png')


# In[38]:


pd.crosstab(df['Spending and Income Cluster'],df['Gender'],normalize='index')


# In[39]:


df.groupby('Spending and Income Cluster')['Age', 'Annual Income (k$)',
       'Spending Score (1-100)'].mean()


# In[40]:


#mulivariate clustering 
from sklearn.preprocessing import StandardScaler


# In[41]:


scale = StandardScaler()


# In[42]:


df.head()


# In[43]:


dff = pd.get_dummies(df,drop_first=True)
dff.head()


# In[44]:


dff.columns


# In[45]:


dff = dff[['Age', 'Annual Income (k$)', 'Spending Score (1-100)','Gender_Male']]
dff.head()


# In[46]:


dff = scale.fit_transform(dff)


# In[47]:


dff = pd.DataFrame(scale.fit_transform(dff))
dff.head()


# In[48]:


intertia_scores3=[]
for i in range(1,11):
    kmeans3=KMeans(n_clusters=i)
    kmeans3.fit(dff)
    intertia_scores3.append(kmeans3.inertia_)
plt.plot(range(1,11),intertia_scores3)


# In[49]:


df


# In[50]:


df.to_csv('Clustering.csv')


# In[ ]:




