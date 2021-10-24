# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:54:26 2021

@author: heeba
"""


import pandas as pd
import numpy as np

testsal = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/SVM/SalaryData_Test (1).csv")
trainsal = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/SVM/SalaryData_Train (1).csv")
testsal.describe()
trainsal.describe()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
testsal["workclass"] = lb.fit_transform(testsal["workclass"])
testsal["education"] = lb.fit_transform(testsal["education"])
testsal["maritalstatus"] = lb.fit_transform(testsal["maritalstatus"])
testsal["occupation"] = lb.fit_transform(testsal["occupation"])
testsal["relationship"] = lb.fit_transform(testsal["relationship"])
testsal["race"] = lb.fit_transform(testsal["race"])
testsal["sex"] = lb.fit_transform(testsal["sex"])
testsal["native"] = lb.fit_transform(testsal["native"])
testsal["Salary"] = lb.fit_transform(testsal["Salary"])

trainsal["workclass"] = lb.fit_transform(trainsal["workclass"])
trainsal["education"] = lb.fit_transform(trainsal["education"])
trainsal["maritalstatus"] = lb.fit_transform(trainsal["maritalstatus"])
trainsal["occupation"] = lb.fit_transform(trainsal["occupation"])
trainsal["relationship"] = lb.fit_transform(trainsal["relationship"])
trainsal["race"] = lb.fit_transform(trainsal["race"])
trainsal["sex"] = lb.fit_transform(trainsal["sex"])
trainsal["native"] = lb.fit_transform(trainsal["native"])
trainsal["Salary"] = lb.fit_transform(trainsal["Salary"])



testsal = testsal.fillna(0)
trainsal = trainsal.fillna(0)
from sklearn.model_selection import train_test_split
train,test = train_test_split(testsal, test_size = 0.20)
testsal.dtypes

train_X = trainsal.iloc[:,:-1]
train_y = trainsal.iloc[:,-1]
test_X  = testsal.iloc[:,:-1]
test_y  = testsal.iloc[:, -1]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
model_linear.fit(test_X, test_y)
pred_saltest_linear = model_linear.predict(test_X)
pred_saltrain_linear = model_linear.predict(train_X)
np.mean(pred_saltest_linear == test_y)
np.mean(pred_saltrain_linear == train_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_saltest_rbf = model_rbf.predict(test_X)
pred_saltrain_rbf = model_rbf.predict(train_X)
np.mean(pred_saltest_rbf==test_y)
np.mean(pred_saltrain_rdf==train_y)




######################################################
########### forestfire ###############################

import pandas as pd
import numpy as np

letters = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/SVM/forestfires.csv")
letters.describe()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
letters["month"] = lb.fit_transform(letters["month"])
letters["day"] = lb.fit_transform(letters["day"])
letters["size_category"] = lb.fit_transform(letters["size_category"])


from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(letters, test_size = 0.20)

train_X = train.iloc[:, 1:]
train_y = train.iloc[:, 0]
test_X  = test.iloc[:, 1:]
test_y  = test.iloc[:, 0]


# kernel = linear
help(SVC)
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)
