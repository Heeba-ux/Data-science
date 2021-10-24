# -*- coding: utf-8 -*-
"""
Created on Sat May 29 00:22:22 2021

@author: heeba
"""


### Multinomial Regression ####
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
mode = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/multinominal reg/mdata.csv")
mode.head(10)
lb = LabelEncoder()
mode["female"] = lb.fit_transform(mode["female"])
mode["ses"] = lb.fit_transform(mode["ses"])
mode["schtyp"] = lb.fit_transform(mode["schtyp"])
mode["prog"] = lb.fit_transform(mode["prog"])
mode["honors"] = lb.fit_transform(mode["honors"])

mode.drop(['Unnamed: 0'],axis =1,inplace=True)
mode.describe()
mode.honors.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "honors", y = "ses", data = mode)
sns.boxplot(x = "honors", y = "prog", data = mode)
sns.boxplot(x = "honors", y = "schtyp", data = mode)
sns.boxplot(x = "honors", y = "female", data = mode)
sns.boxplot(x = "honors", y = "read", data = mode)
sns.boxplot(x = "honors", y = "write", data = mode)
sns.boxplot(x = "honors", y = "science", data = mode)


# Scatter plot for each categorical choice of car
sns.stripplot(x = "honors", y = "id", jitter = True, data = mode)
sns.stripplot(x = "honors", y = "ses", jitter = True, data = mode)
sns.stripplot(x = "honors", y = "female", jitter = True, data = mode)
sns.stripplot(x = "honors", y = "prog", jitter = True, data = mode)
sns.stripplot(x = "honors", y = "read", jitter = True, data = mode)
sns.stripplot(x = "honors", y = "write", jitter = True, data = mode)
sns.stripplot(x = "honors", y = "math", jitter = True, data = mode)

# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "honors") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()

train, test = train_test_split(mode, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
test_predict
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
train_predict
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
################################################################################
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
mode = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/multinominal reg/loan.csv")
mode.head(10)
mode.columns
mode.drop(['emptitle','url','id','memberid','term','subgrade','lastpymntd','lastcreditpulld','issued','earliestcrline','pymntplan','desc','title','emplength','zipcode','addrstate'], axis = 1,inplace = True)
mode.isna().sum()

# To drop NaN values
mode = mode.dropna()

lb = LabelEncoder()
mode["grade"] = lb.fit_transform(mode["grade"])
mode["homeownership"] = lb.fit_transform(mode["homeownership"])
mode["verificationstatus"] = lb.fit_transform(mode["verificationstatus"])
mode["loanstatus"] = lb.fit_transform(mode["loanstatus"])
mode["purpose"] = lb.fit_transform(mode["purpose"])
mode["initialliststatus"] = lb.fit_transform(mode["initialliststatus"])


mode['intrate'] = mode['intrate'].str.replace("%","")
mode['revolutil'] = mode['revolutil'].str.replace("%","")

mode.drop(['Unnamed: 0'],axis =1,inplace=True)
mode.describe()
mode.loanstatus.value_counts()

# Boxplot of independent variable distribution for each category of choice 
sns.boxplot(x = "loanstatus", y = "totalpymnt", data = mode)
sns.boxplot(x = "loanstatus", y = "lastpymntamnt", data = mode)
sns.boxplot(x = "loanstatus", y = "homeownership", data = mode)
sns.boxplot(x = "loanstatus", y = "installment", data = mode)
sns.boxplot(x = "loanstatus", y = "loanamnt", data = mode)



# Scatter plot for each categorical choice of car
sns.stripplot(x = "loanstatus", y = "totalpymnt", jitter = True, data = mode)
sns.stripplot(x = "loanstatus", y = "lastpymntamnt", jitter = True, data = mode)
sns.stripplot(x = "loanstatus", y = "homeownership", jitter = True, data = mode)
sns.stripplot(x = "loanstatus", y = "installment", jitter = True, data = mode)
sns.stripplot(x = "loanstatus", y = "loanamnt", jitter = True, data = mode)


# Scatter plot between each possible pair of independent variable and also histogram for each independent variable 
sns.pairplot(mode) # Normal
sns.pairplot(mode, hue = "loanstatus") # With showing the category of each car choice in the scatter plot

# Correlation values between each independent features
mode.corr()

train, test = train_test_split(mode, test_size = 0.2)

# ‘multinomial’ option is supported only by the ‘lbfgs’ and ‘newton-cg’ solvers
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:, 0])
help(LogisticRegression)

test_predict = model.predict(test.iloc[:, 1:]) # Test predictions
test_predict
# Test accuracy 
accuracy_score(test.iloc[:,0], test_predict)

train_predict = model.predict(train.iloc[:, 1:]) # Train predictions 
train_predict
# Train accuracy 
accuracy_score(train.iloc[:,0], train_predict) 
