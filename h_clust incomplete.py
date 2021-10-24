# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 22:28:13 2021

@author: heeba
"""


import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import seaborn as sns
df = pd.read_excel("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/hclust/Telco_Customer_Churn.xlsx",encoding='latin1')
df.describe()
df.info()
df.shape
plt.figure(1), plt.subplot(121),
sns.distplot(df['Tenure in Months']);

plt.figure(1), plt.subplot(121), 
sns.distplot(df['Monthly Charge']);

plt.figure(1), plt.subplot(121), 
sns.distplot(df['Total Charges']);

df.drop(["Customer ID"], axis=1)
df.info()
from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()
X = df.iloc[:, 0:28]
y = df['Total Revenue']
y=df.iloc[:,28:]
df.columns

X['Quarter']= labelencoder.fit_transform(X['Quarter'])
X['Referred a Friend'] = labelencoder.fit_transform(X['Referred a Friend'])
X['Offer'] = labelencoder.fit_transform(X['Offer'])
X['Phone Service'] = labelencoder.fit_transform(X['Phone Service'])
X['Multiple Lines'] = labelencoder.fit_transform(X['Multiple Lines'])
X['Internet Service'] = labelencoder.fit_transform(X['Internet Service'])
X['Internet Type'] = labelencoder.fit_transform(X['Internet Type'])
X['Online Security'] = labelencoder.fit_transform(X['Online Security'])
X['Online Backup'] = labelencoder.fit_transform(X['Online Backup'])
X['Device Protection Plan'] = labelencoder.fit_transform(X['Device Protection Plan'])
X['Premium Tech Support'] = labelencoder.fit_transform(X['Premium Tech Support'])
X['Streaming TV'] = labelencoder.fit_transform(X['Streaming TV'])
X['Streaming Movies'] = labelencoder.fit_transform(X['Streaming Movies'])
X['Streaming Music'] = labelencoder.fit_transform(X['Streaming Music'])
X['Unlimited Data'] = labelencoder.fit_transform(X['Unlimited Data'])
X['Contract'] = labelencoder.fit_transform(X['Contract'])
X['Paperless Billing'] = labelencoder.fit_transform(X['Paperless Billing'])
X['Payment Method'] = labelencoder.fit_transform(X['Payment Method'])


y = labelencoder.fit_transform(y)
y = pd.DataFrame(y)

### we have to convert y to data frame so that we can use concatenate function

# concatenate X and y

df_new = pd.concat([X, y], axis =1)
## rename column name
df_new.columns
df_new = df_new.rename(columns={0:'Type'})
df_new

pip install gower
import gower
gower.gower_matrix(X)
gower.gower_matrix(df_new)

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

np.all(np.isfinite(df_new))
pd.DataFrame(df_new).fillna(1)
df_new
df.isna().sum()
df_norm.columns[df_norm.nunique() == 1]
df_norm.columns
df_norm.dtypes
df_norm = df_norm.drop(['Gender_F', 'Gender_M','Marital Status_Divorced', 'Marital Status_Married','Marital Status_Single','Education_Bachelor',
       'Education_College', 'Education_Doctor',
       'Education_High School or Below', 'Education_Master','Marital Status_Divorced', 'Marital Status_Married',
       'Marital Status_Single','Policy_Corporate L1', 'Policy_Corporate L2', 'Policy_Corporate L3',
       'Policy_Personal L1', 'Policy_Personal L2', 'Policy_Personal L3',
       'Policy_Special L1', 'Policy_Special L2', 'Policy_Special L3','Vehicle Class_Luxury SUV', 'Vehicle Class_SUV',
       'Vehicle Class_Sports Car', 'Vehicle Class_Two-Door Car'],axis=1)
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method="complete",metric="euclidean")
h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

Univ['clust'] = cluster_labels # creating a new column and assigning it to new column 

Univ1 = Univ.iloc[:, [7,0,1,2,3,4,5,6]]
Univ1.head()

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()
df_norm = norm_func(df_new)
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 
z = linkage(df_norm, method = "complete", metric = "euclidean")
df_norm = norm_func(df.iloc[:, 1:])
df_norm.describe()








df = pd.get_dummies(df.columns =[‘Quarter’,‘Referred a Friend’,‘Offer’,’Phone Service’,’Multiple Lines’,’Internet Service’,‘Internet Type’,’Online Backup’,'Online Security’,'Device Protection Plan’,’Premium Tech Support’,’Streaming TV’,‘Streaming Movies’,’Streaming Music’,’Unlimited Data’,’Contract’,’Paperless Billing’,’Payment Method’],drop_first = True)

trainDfDummies = pd.get_dummies(trainDf, columns=['Col1', 'Col2', 'Col3', 'Col4'])
df = pd.get_dummies(df.iloc[:, 1:22])
df = pd.get_dummies(df.iloc[:, 'col2':'col3':'col5':'col6':'col8':'col9':'col10':'col12':'col13':'col14':'col15':'col16':'col17':'col18':'col19':'col20':'col21':'col22'])
df.describe()

def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(df.iloc[:, 0:])
df_norm.describe()
df.isna().sum()
df.columns[df.nunique() == 1]
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")
