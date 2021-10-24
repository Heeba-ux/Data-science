# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:17:33 2021

@author: DELL
"""


import pandas as pd
import matplotlib.pylab as plt

df_raw = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/hclust/AutoInsurance.csv",encoding='latin1')
df = df_raw

df.describe()
df.info()

df = df.drop(["Customer"], axis=1)

##################  creating Dummy variables using dummies ###############
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df.columns
df.shape
# drop emp_name column
df.drop(['Customer Lifetime Value','Effective To Date','Income','Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies','Total Claim Amount'],axis =1,inplace=True)
df.dtypes


######################################
# Create dummy variables on categorcal columns

df_new = pd.get_dummies(df)

df1 = df_raw
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

df_raw['clust'] = cluster_labels # creating a new column and assigning it to new column 
df_raw.columns
df_raw.shape


df_raw = df_raw.iloc[:, [1,0,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
df_raw.head()

df_raw.columns

# Aggregate mean of each cluster
df_raw.iloc[:, 3:].groupby(df_raw.clust).mean()

# creating a csv file 
df.to_csv("hclustering.csv", encoding = "utf-8")

import os
os.getcwd()
