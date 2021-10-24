# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 14:45:40 2021

@author: Heeba
"""
#############################Eastwestairlines#####################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
df_raw = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/k-means/EastWestAirlines (1).csv",encoding='latin1')
df = df_raw

df.describe()
df.info()
df.columns
df = df.drop(["ID"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 1:])
TWSS = []
k = list(range(2, 12))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column 

df.head()
df_norm.head()
df = df.iloc[:,[7,0,1,2,3,4,5,6]]
df.head()

df.iloc[:, 2:12].groupby(df.clust).mean()

##########################################################################
##############crime_data###############################################

import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
df_raw = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/k-means/crime_data (1).csv",encoding='latin1')
df = df_raw

df.describe()
df.info()
df.columns
df = df.drop(["Unnamed: 0"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 1:])
TWSS = []
k = list(range(2, 4))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column 

df.head()
df_norm.head()
df = df.iloc[:,[4,0,1,2,3]]
df.head()

df.iloc[:, 2:4].groupby(df.clust).mean()
############################################################################
##########################insurance data####################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
df_insurance = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/k-means/Insurance Dataset.csv",encoding='latin1')
df = df_insurance

df.describe()
df.info()
df.columns
df = df.drop(["Claims made"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 1:])
TWSS = []
k = list(range(1, 4))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column 

df.head()
df_norm.head()
df = df.iloc[:,[4,0,1,2,3]]
df.head()

df.iloc[:, 2:4].groupby(df.clust).mean()
#######################################################################
##########################telco customer churn#########################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
df_telco = pd.read_excel("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/k-means/Telco_customer_churn (1).xlsx",encoding='latin1')
df = df_telco

df.describe()
df.info()
df.columns
df = df.drop(["Customer ID","Quarter","Referred a Friend","Offer","Phone Service","Multiple Lines","Internet Service","Internet Type","Online Security","Online Backup","Device Protection Plan","Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data","Contract","Paperless Billing","Payment Method"], axis=1)

# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 1:])
df_norm.describe()
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
TWSS = []
k = list(range(2, 54))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column 

df.head()
df_norm.head()

df = df.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
df.head()

df.iloc[:, 2:54].groupby(df.clust).mean()

df.to_csv("Kmeans_university.csv", encoding = "utf-8")

import os
os.getcwd()
###########################################################################
###########################autoinsurance###################################
import pandas as pd
import matplotlib.pylab as plt
from sklearn.cluster import	KMeans
df_auto = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/k-means/AutoInsurance (1).csv",encoding='latin1')
df = df_auto

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

df1 = df_auto
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
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 3)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
df['clust'] = mb # creating a  new column and assigning it to new column 

df.head()
df_norm.head()

df = df.iloc[:,[9,0,1,2,3,4,5,6,7,8]]
df.head()

df.iloc[:, 0:9].groupby(df.clust).mean()


