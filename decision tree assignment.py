# -*- coding: utf-8 -*-
"""
Created on Mon May 24 00:52:46 2021

@author: heeba
"""
#######################################company data###########################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/Company_Data.csv")

data.isnull().sum()
data.dropna()
data.columns

#converting into binary
lb = LabelEncoder()
data["ShelveLoc"] = lb.fit_transform(data["ShelveLoc"])
data["Urban"] = lb.fit_transform(data["Urban"])
data["US"] = lb.fit_transform(data["US"])

#discretizing the data

data.Sales.describe()
data['class_variable'] = pd.cut(x=data['Sales'], bins=[-1, 8, 50], labels=['0', '1'])
data['class_variable'].unique()

#data["default"]=lb.fit_transform(data["default"])

data['class_variable'].value_counts()
colnames = list(data.columns)

predictors = colnames[:11]
target = colnames[11]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#no pruning required, good accuracy and no false negatives
###################################################################
#######################diabetes####################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/Diabetes.csv")

data.isnull().sum()
data.dropna()
data.columns

data.rename(columns={' Class.variable':'class_variable'}, inplace=True)
data.columns

#converting into binary
lb = LabelEncoder()
data["class_variable"] = lb.fit_transform(data["class_variable"])


data['class_variable'].unique()

data['class_variable'].value_counts()
colnames = list(data.columns)

predictors = colnames[:8]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#pruning required as the model is underfit

#PRUNING

df = data
df.info()

# Dummy variables
df.head()

# Input and Output Split
predictors = df.loc[:, df.columns!="class_variable"]
type(predictors)

target = df["class_variable"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.2, random_state=0)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)


# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
r2_score(y_test, test_pred2)

# Error on train dataset
mean_squared_error(y_train, train_pred2)
r2_score(y_train, train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
r2_score(y_train, train_pred3)

######################################################################
#############################fraud#######################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/Fraud_check.csv")

data.isnull().sum()
data.dropna()
data.columns

#converting into binary
lb = LabelEncoder()
data["Undergrad"] = lb.fit_transform(data["Undergrad"])
data["MaritalStatus"] = lb.fit_transform(data["MaritalStatus"])
data["Urban"] = lb.fit_transform(data["Urban"])


#discretizing the salary

data.describe()
data['class_variable'] = pd.cut(x=data['Taxable.Income'], bins=[0, 30000, 100000], labels=['0', '1'])
data['class_variable'].unique()

#data["default"]=lb.fit_transform(data["default"])

data['class_variable'].value_counts()
colnames = list(data.columns)

predictors = colnames[:6]
target = colnames[6]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

#test pruning as model too good to be true

#PRUNING

df = data
df.info()


# Dummy variables
df.head()

# Input and Output Split
predictors = df.loc[:, df.columns!="class_variable"]
type(predictors)

target = df["class_variable"]
type(target)

# Train Test partition of the data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(predictors, target, test_size = 0.3, random_state=0)

# Train the Regression DT
from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 10)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Measuring accuracy
from sklearn.metrics import mean_squared_error, r2_score

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)

# Pruning the Tree
# Minimum observations at the internal node approach
regtree2 = tree.DecisionTreeRegressor(min_samples_split = 3)
regtree2.fit(x_train, y_train)

# Prediction
test_pred2 = regtree2.predict(x_test)
train_pred2 = regtree2.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred2)
r2_score(y_test, test_pred2)

# Error on train dataset
mean_squared_error(y_train, train_pred2)
r2_score(y_train, train_pred2)

## Minimum observations at the leaf node approach
regtree3 = tree.DecisionTreeRegressor(min_samples_leaf = 3)
regtree3.fit(x_train, y_train)

# Prediction
test_pred3 = regtree3.predict(x_test)
train_pred3 = regtree3.predict(x_train)

# measure of error on test dataset
mean_squared_error(y_test, test_pred3)
r2_score(y_test, test_pred3)

# measure of error on train dataset
mean_squared_error(y_train, train_pred3)
r2_score(y_train, train_pred3)
##############################################################################
################################HR##############################################
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/HR_DT.csv")

data.isnull().sum()
data.dropna()
data.columns
data.dtypes

#converting into binary
lb = LabelEncoder()
data["Position of the employee"] = lb.fit_transform(data["Position of the employee"])


#classifying the class variable

data[" monthly income of employee"].describe()
data["no of Years of Experience of employee"].describe()

import matplotlib.pyplot as plt

plt.scatter(data[" monthly income of employee"],data["no of Years of Experience of employee"])
plt.show()


# create a list of our conditions
conditions = [
    (data[' monthly income of employee'] >100000) 
    ]

# create a list of the values we want to assign for each condition
values = ['0']

# create a new column and use np.select to assign values to it using our lists as arguments
data['class_variable'] = np.select(conditions, values, default=1)



# display updated DataFrame
data.head()
    
data['class_variable'].unique()

#data["default"]=lb.fit_transform(data["default"])

data['class_variable'].value_counts()
colnames = list(data.columns)

predictors = colnames[:3]
target = colnames[3]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.3)

from sklearn.tree import DecisionTreeClassifier as DT

help(DT)
model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

