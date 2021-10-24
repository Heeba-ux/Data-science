# -*- coding: utf-8 -*-
"""
Created on Fri May 28 10:39:32 2021

@author: heeba
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

#Importing Data
claimants = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/logistic reg/Affairs.csv", sep = ",")
claimants.columns
#removing CASENUM
c1 = claimants.drop('Unnamed: 0', axis = 1)
c1.head(11)
c1.describe()
c1.isna().sum()

# To drop NaN values
df = claimants.dropna()

# Imputating the missing values           
# Mean Imputation - CLMAGE is a continuous data

mean_value = c1.naffairs.mean()
mean_value
c1.naffairs = c1.naffairs.fillna(mean_value)
c1.naffairs.isna().sum()

# For Median imputation try this
# median_value = claimants.CLMAGE.median()
# claimants1['CLMAGE'] = claimants1['CLMAGE'].fillna(median_value)
c1['naffairs'] = np.where(c1['naffairs'] == 0, 0 ,1)

c1.naffairs.mode()
# For Mode - for Discrete variables
# CLMSEX
mode_naffairs = c1.naffairs.mode()
mode_naffairs
c1.naffairs = c1.naffairs.fillna((mode_naffairs)[0])
c1.naffairs.isna().sum()

# Alternate approach
########## Median Imputation for all the columns ############
c1.fillna(c1.median(), inplace=True)
c1.isna().sum()

c1.naffairs.median()

#############################################################

# Model building 
# import statsmodels.formula.api as sm

logit_model = sm.logit('naffairs ~ kids + vryunhap + yrsmarr4 + yrsmarr5 + yrsmarr6', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, 1: ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.naffairs, pred)
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
c1["pred"] = np.zeros(601)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["naffairs"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(181)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (33 + 43)/(181) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(420)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = (178 + 77)/(420)
print(accuracy_train)

###########################################################################
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
claimants = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/logistic reg/advertising.csv", sep = ",")
claimants.columns
#removing CASENUM
c1 = claimants.drop(columns=['Ad_Topic_Line','City','Country','Timestamp'], axis = 1)
c1 = claimants
c1.columns
c1.head(11)
c1.describe()
c1.isna().sum()

# To drop NaN values
df = claimants.dropna()
#############################################################

# Model building 
# import statsmodels.formula.api as sm
c1.rename(columns={'Daily Internet Usage':'D','Clicked_on_Ad':'coa','Daily_Time_ Spent _on_Site':'dts'}, inplace=True)
logit_model = sm.logit('coa ~ Age + dts + Area_Income + D + Male', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, :6 ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.coa, pred)
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
c1["pred"] = np.zeros(1000)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["coa"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('coa ~ Age + dts + Area_Income + D + Male', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(300)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['coa'])
confusion_matrix

accuracy_test = (146 + 140)/(300) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["coa"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["coa"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :6 ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(700)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['coa'])
confusion_matrx

accuracy_train = (348 + 339)/(700)
print(accuracy_train)
###################################################################################
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
claimants = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/logistic reg/election_data.csv", sep = ",")
claimants.columns
#removing CASENUM
#c1= claimants.drop(columns=['Ad_Topic_Line','City','Country','Timestamp'], axis = 1)
c1 = claimants
c1.columns
c1.head(11)
c1.describe()
c1.isna().sum()

# To drop NaN values
c1 = claimants.dropna()

# Imputating the missing values           
# Mean Imputation - CLMAGE is a continuous data
# For Median imputation try this
# median_value = claimants.CLMAGE.median()
# claimants1['CLMAGE'] = claimants1['CLMAGE'].fillna(median_value)
#1['id'] = np.where(c1['id'] == 0, 0 ,1)

mode_Result = c1.Result.mode()
mode_Result
c1.Result = c1.Result.fillna((mode_Result)[0])
c1.Result.isna().sum()

# CLMINSUR
mode_id = c1['id'].mode()
mode_id
c1['id'] = c1['id'].fillna((mode_id)[0])
c1.id.isna().sum()


##################################

# Model building 
# import statsmodels.formula.api as sm
c1.rename(columns={'Popularity Rank':'rank','Amount Spent':'amt','Election-id':'id'}, inplace=True)
c1.columns
logit_model = sm.logit('Result ~ id + Year + amt', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, : ])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.Result, pred)
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
c1["pred"] = np.zeros(10)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["Result"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('Result ~ id + Year + amt', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(3)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

accuracy_test = (1+ 2/(3) )
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Result])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(7)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx

accuracy_train = (2 + 3)/(7)
print(accuracy_train)
#####################################################################################33
import pandas as pd
import numpy as np
# import seaborn as sb
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split # train and test 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Importing Data
claimants = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/logistic reg/bank_data.csv", sep = ",")
claimants.columns
#removing CASENUM
#c1= claimants.drop(columns=['Ad_Topic_Line','City','Country','Timestamp'], axis = 1)
c1 = claimants
c1.columns
c1.head(11)
c1.describe()
c1.isna().sum()

# To drop NaN values
c1 = claimants.dropna()

# Imputating the missing values           
# Mean Imputation - CLMAGE is a continuous data
# For Median imputation try this
# median_value = claimants.CLMAGE.median()
# claimants1['CLMAGE'] = claimants1['CLMAGE'].fillna(median_value)
#1['id'] = np.where(c1['id'] == 0, 0 ,1)

mode_y = c1.y.mode()
mode_y
c1.y = c1.y.fillna((mode_y)[0])
c1.y.isna().sum()


##################################

# Model building 
# import statsmodels.formula.api as sm
#c1.rename(columns={'Popularity Rank':'rank','Amount Spent':'amt','Election-id':'id'}, inplace=True)
c1.columns
logit_model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + concellular + contelephone + conunknown + divorced + married + single + joadmin + jobluecollar + joentrepreneur + johousemaid + jomanagement + joretired + jounemployed + jounknown', data = c1).fit()

#summary
logit_model.summary2() # for AIC
logit_model.summary()

pred = logit_model.predict(c1.iloc[ :, : 32])

# from sklearn import metrics
fpr, tpr, thresholds = roc_curve(c1.y, pred)
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
c1["pred"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
c1.loc[pred > optimal_threshold, "pred"] = 1
# classification report
classification = classification_report(c1["pred"], c1["y"])
classification


### Splitting the data into train and test data 
# from sklearn.model_selection import train_test_split
train_data, test_data = train_test_split(c1, test_size = 0.3) # 30% test data

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + pdays + previous + poutfailure + poutother + poutsuccess + poutunknown + concellular + contelephone + conunknown + divorced + married + single + joadmin + jobluecollar + joentrepreneur + johousemaid + jomanagement + joretired + jounemployed + jounknown', data = train_data).fit()

#summary
model.summary2() # for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(13564)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (9710 + 1345/(34) )
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test


# prediction on train data
train_pred = model.predict(train_data.iloc[ :, :])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(31647)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (22770 + 2965)/(34)
print(accuracy_train)
