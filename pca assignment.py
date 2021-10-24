# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 19:55:48 2021

@author: DELL
"""


import pandas as pd 
import numpy as np

data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/wine.csv",encoding='latin1')
data.describe()
uni = data.drop(["Type"], axis = 1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale 

# Considering only numerical data 
uni.data = uni.iloc[:, 1:]

# Normalizing the numerical data 
uni_normal = scale(uni.data)
uni_normal

pca = PCA(n_components = 12)
pca_values = pca.fit_transform(uni_normal)

# The amount of variance that each PCA explains is 
var = pca.explained_variance_ratio_
var

pca.components_
pca.components_[0]
# Cumulative variance 

var1 = np.cumsum(np.round(var, decimals = 4) * 100)
var1

# Variance plot for PCA components obtained 
plt.plot(var1, color = "red")

# PCA scores
pca_values

pca_data = pd.DataFrame(pca_values)
pca_data.columns = "comp0", "comp1", "comp2", "comp3", "comp4", "comp5", "comp6", "comp7", "comp8", "comp9", "comp10", "comp11"
final = pd.concat([uni.data, pca_data.iloc[:, 0:3]], axis = 1)
import matplotlib.pylab as plt
plt.scatter(x = final.comp0, y = final.comp1)
final.describe()
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(uni_normal, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(15, 8));plt.title('PCA H_Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z, 
    leaf_rotation = 0,  # rotates the x axis labels
    leaf_font_size = 10 # font size for the x axis labels
)
plt.show()

# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 3, linkage = 'complete', affinity = "euclidean").fit(uni_normal) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

data['clust'] = cluster_labels # creating a new column and assigning it to new column 

data1 = data.iloc[:, [7,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
data1.head()

# Aggregate mean of each cluster
data1.iloc[:, 2:].groupby(data1.clust).mean()
# Scatter diagram
import matplotlib.pylab as plt
plt.scatter(x = data1.Alcohol, y = data1.Malic)
