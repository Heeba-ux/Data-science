# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 10:18:40 2021

@author: Heeba
"""


import pandas as pd
import matplotlib.pylab as plt

df_raw = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/hclust/EastWestAirlines 1.xlsx",encoding='latin1')
df = df_raw

df.describe()
df.info()
df.columns
df = df.drop(["ID#"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['clust'] = cluster_labels # creating a new column and assigning it to new column 

df = df.iloc[:, [7,0,1,2,3,4,5,6]]
df.head()

# Aggregate mean of each cluster
df.iloc[:, 2:].groupby(df.clust).mean()

# creating a csv file 
df.to_csv("hclustering.csv", encoding = "utf-8")

import os
os.getcwd()


#######################################################
###############crime_data##############################

import pandas as pd
import matplotlib.pylab as plt

Univ1 = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/hclust/crime_data.csv",encoding='latin1')

Univ1.describe()
Univ1.info()

Univ = Univ1.drop(["Unnamed: 0"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Univ.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

Univ['clust'] = cluster_labels # creating a new column and assigning it to new column 

Univ1 = Univ.iloc[:,[5,0,1,2,3,4]]
Univ1.head()
# Aggregate mean of each cluster
Univ1.iloc[:, 1:].groupby(Univ1.clust).mean()

# creating a csv file 
Univ1.to_csv("crime_data.csv", encoding = "utf-8")

import os
os.getcwd()

############################################################33
############### telco_churn_data############################
import pandas as pd
import matplotlib.pylab as plt

df_raw = pd.read_excel("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/hclust/Telco_customer_churn.xlsx")
df = df_raw

df.describe()
df.info()

df = df.drop(["Customer ID","Quarter","Referred a Friend","Offer","Phone Service","Multiple Lines","Internet Service","Internet Type","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data","Contract","Paperless Billing","Payment Method"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 1:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import scipy.cluster.hierarchy as sch 
import numpy as np

import gower

dm = gower.gower_matrix(df)
dm = pd.DataFrame(dm)

upper = dm.where(np.triu(np.ones(dm.shape), k=1).astype(np.bool)).stack().to_numpy()


Zd = linkage(upper) 

k = fcluster(Zd, 3, criterion='maxclust')
k

dendrogram(Zd)

# Dendrogram
plt.figure(c=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(k, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# # Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "gower").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

df['clust'] = cluster_labels # creating a new column and assigning it to new column 

df = df.iloc[:, [7,0,1,2,3,4,5,6]]
df.head()

# # Aggregate mean of each cluster
df.iloc[:, 2:].groupby(df.clust).mean()

# creating a csv file 
df.to_csv("hclustering.csv", encoding = "utf-8")

import os
os.getcwd()

#########################################################################
########################auto insurance###################################
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

data_raw = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/hclust/AutoInsurance.csv",encoding='latin1')
df = data_raw

df.describe()
df.info()

df = df.drop(["Customer"], axis=1)
df.columns
df.shape
df.drop(['Customer Lifetime Value','Effective To Date','Income','Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies','Total Claim Amount'],axis =1,inplace=True)
df.dtypes

df_new = pd.get_dummies(df)

df1 = data_raw
df1.columns
df1.dtypes
df1 = df1.drop(["Customer",'State','Response', 'Coverage',
       'Education', 'Effective To Date', 'EmploymentStatus', 'Gender','Location Code', 'Marital Status','Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel','Vehicle Class', 'Vehicle Size'], axis=1)

df1.dtypes

df_new = pd.concat([df1, df_new], axis =1)
df_new.dtypes
# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df_new.iloc[:, 0:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 8, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data_raw['clust'] = cluster_labels # creating a new column and assigning it to new column 
data_raw.columns
data_raw.shape


data_raw = df_raw.iloc[:, [1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
data_raw.head()

data_raw.columns

# Aggregate mean of each cluster
df_raw.iloc[:, 3:].groupby(df_raw.clust).mean()

# creating a csv file 
df.to_csv("hclustering.csv", encoding = "utf-8")

import os
os.getcwd()
