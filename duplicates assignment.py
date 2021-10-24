# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 21:39:23 2021

@author: DELL
"""


import pandas as pd
df= pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/OnlineRetail.csv",encoding='latin1')
df
df.describe()
df.dtypes

df.UnitPrice = df.UnitPrice.astype('int64') 
df.dtypes
df.Quantity = df.Quantity.astype('object') 
df.dtypes
df
df.CustomerID = df.CustomerID.astype('float64') 
df.dtypes
df

duplicate = df.duplicated()
duplicate
sum(duplicate)

data1 = df.drop_duplicates() 
data1

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.offline import plot
import plotly.graph_objs as go
plt.style.use("ggplot")

plt.bar(height = data1.Country, x = np.arange(1,501,10))
plt.bar

plt.hist(data1.Country.head(500))
plt.boxplot(data1.Quantity.head(500))

