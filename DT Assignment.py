# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:20:43 2021

@author: Heeba
"""
########################################company data######################3
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/Company_Data.csv")
data.ShelveLoc = data.ShelveLoc.replace( {"Good":1 , "Medium":2 , "Bad":3})
data.Urban = data.Urban.replace( {"Yes":1 , "No":2})
data.US = data.US.replace( {"Yes":1 , "No":2})

data.columns
data.describe
data.dtypes
colnames = list(data.columns)

predictors = colnames[:1]
target = colnames[1]

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
from sklearn import tree
Dtree.plot_tree(model)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 30))
sns.heatmap(data.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
################################# Diabetes ##################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

df = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/Diabetes.csv")
df.head()

pd.set_option("display.float_format", "{:.2f}".format)
df.describe()
df.columns
df.dtypes
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df[" outcome"] = lb.fit_transform(df[" outcome"])

colnames = list(df.columns)

predictors = colnames[:9]
target = colnames[8]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3)

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
from sklearn import tree
tree.plot_tree(model)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 30))
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
################################# fraud #################################
import pandas as pd
import numpy as np
import matplotlib.pyplot

fraud = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/Fraud_check.csv")

##Converting the Taxable income variable to bucketing. 
fraud["income"]="<=30000"
fraud.loc[fraud["TaxableIncome"]>=30000,"income"]="Good"
fraud.loc[fraud["TaxableIncome"]<=30000,"income"]="Risky"

##Droping the Taxable income variable
fraud.drop(["TaxableIncome"],axis=1,inplace=True)

fraud.rename(columns={"Undergrad":"undergrad","MaritalStatus":"marital","CityPopulation":"population","WorkExperience":"experience","Urban":"urban"},inplace=True)
## As we are getting error as "ValueError: could not convert string to float: 'YES'".
## Model.fit doesnt not consider String. So, we encode

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in fraud.columns:
    if fraud[column_name].dtype == object:
        fraud[column_name] = le.fit_transform(fraud[column_name])
    else:
        pass
  
##Splitting the data into featuers and labels
features = fraud.iloc[:,0:5]
labels = fraud.iloc[:,5]

## Collecting the column names
colnames = list(fraud.columns)
predictors = colnames[0:5]
target = colnames[5]
##Splitting the data into train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)

##Model building
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)
model.estimators_
model.classes_
model.n_features_
model.n_classes_

model.n_outputs_

model.oob_score_


##Predictions on train data
prediction = model.predict(x_train)

##Accuracy
# For accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
np.mean(prediction == y_train)


##Confusion matrix
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)
confusion
##Prediction on test data
pred_test = model.predict(x_test)
pred_test
##Accuracy
acc_test =accuracy_score(y_test,pred_test)
acc_test
## In random forest we can plot a Decision tree present in Random forest
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 30))
sns.heatmap(fraud.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
############################## HR #####################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
sns.set_style("whitegrid")
plt.style.use("fivethirtyeight")

df = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/decision tree/HR_DT.csv")
df.head()

pd.set_option("display.float_format", "{:.2f}".format)
df.describe()
df.columns
df.dtypes
from sklearn.preprocessing import LabelEncoder

df.drop(['Position of the employee'], axis='columns', inplace=True)
colnames = list(df.columns)
df.rename(columns={"no of Years of Experience of employee":"experienece"," monthly income of employee":"income"},inplace=True)
plt.hist(df.experienece)

fig = plt.figure(figsize = (10, 5))
 
dataFrame = pd.DataFrame(df);


# Draw a vertical bar chart

dataFrame.plot.bar(x="income[1:10]", y="experienece[1:10]", rot=50, title="HR RECRUITMENT");
plt.bar(df["experienece"], df["income"])
plt.show(block=True);

predictors = colnames[:1]
target = colnames[1]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.3)

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
from sklearn import tree
tree.plot_tree(model)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(30, 30))
sns.heatmap(df.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":15})
