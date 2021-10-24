# -*- coding: utf-8 -*-
"""
Created on Fri Sep 24 11:19:33 2021

@author: DELL
"""


import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
df = pd.read_csv("C:/Users/DELL/Downloads/reg.csv")
df.columns

df.isna().sum()

df[' Age']
logit_model = sm.logit('Attrition ~ StockOptionLevel + Department + JobRole + PercentSalaryHike', data = df).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(df.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(df.Attrition, pred)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold


import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]


# Plot tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)


# filling all the cells with zeroes
df["pred"] = np.zeros(1470)
# taking threshold value and above the prob value will be treated as correct value 
df.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(df["pred"], df["Attrition"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(df, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Attrition ~ StockOptionLevel + Department + JobRole + PercentSalaryHike', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(441)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Attrition'])
confusion_matrix

accuracy_test = (247 + 121)/(441) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Attrition"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Attrition"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :32 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(1029)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Attrition'])
confusion_matrx

accuracy_train = (612 + 253)/(1029)
print(accuracy_train)

print(accuracy_test)
