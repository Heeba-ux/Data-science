# -*- coding: utf-8 -*-
"""
Created on Sat May 29 19:49:02 2021

@author: DELL
"""


# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/lasso ridge/50_Startups (1).csv")

# Rearrange the order of the variables
#car = car.iloc[:, [1, 0, 2, 3, 4]]
car.columns
car.drop(['State'],axis =1,inplace=True)
# Correlation matrix 
a = car.corr()
a
car.rename(columns={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms','Profit':'p'}, inplace=True)
# EDA
a1 = car.describe()

# Sctter plot and histogram between variables
sns.pairplot(car) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols("p ~ rd + ad + ms", data = car).fit()
model_train.summary()

# Prediction
pred = model_train.predict(car)
# Error
resid  = pred - car.p
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car.iloc[:, 1:], car.p)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car.iloc[:, 1:])

# Adjusted r-square
lasso.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((pred_lasso - car.p)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car.iloc[:, 1:], car.p)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car.columns[1:]))

rm.alpha

pred_rm = rm.predict(car.iloc[:, 1:])

# Adjusted r-square
rm.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((pred_rm - car.p)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car.iloc[:, 1:], car.p) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car.columns[1:]))

enet.alpha

pred_enet = enet.predict(car.iloc[:, 1:])

# Adjusted r-square
enet.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((pred_enet - car.p)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car.iloc[:, 1:], car.p)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((lasso_pred - car.p)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car.iloc[:, 1:], car.p)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((ridge_pred - car.p)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car.iloc[:, 1:], car.p)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((enet_pred - car.p)**2))
##########################################################################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/lasso ridge/Computer_Data (1).csv")

# Rearrange the order of the variables
#car = car.iloc[:, [1, 0, 2, 3, 4]]
car.columns
car = car.drop(columns=['Unnamed: 0', 'cd', 'multi', 'premium'])

car.rename(columns={'price':'p','speed':'s','Marketing Spend':'ms','Profit':'p'}, inplace=True)
# Correlation matrix 
a = car.corr()
a

# EDA
a1 = car.describe()

# Sctter plot and histogram between variables
sns.pairplot(car) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols("p ~ s + hd + ram + screen + ads + trend", data = car).fit()
model_train.summary()

# Prediction
pred = model_train.predict(car)
# Error
resid  = pred - car.p
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car.iloc[:, 1:], car.p)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car.iloc[:, 1:])

# Adjusted r-square
lasso.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((pred_lasso - car.p)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car.iloc[:, 1:], car.p)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car.columns[1:]))

rm.alpha

pred_rm = rm.predict(car.iloc[:, 1:])

# Adjusted r-square
rm.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((pred_rm - car.p)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car.iloc[:, 1:], car.p) 

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car.columns[1:]))

enet.alpha

pred_enet = enet.predict(car.iloc[:, 1:])

# Adjusted r-square
enet.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((pred_enet - car.p)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car.iloc[:, 1:], car.p)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((lasso_pred - car.p)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car.iloc[:, 1:], car.p)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((ridge_pred - car.p)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car.iloc[:, 1:], car.p)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car.iloc[:, 1:], car.p)

# RMSE
np.sqrt(np.mean((enet_pred - car.p)**2))
#################################################################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/lasso ridge/ToyotaCorolla (1).csv",encoding='latin1')

# Rearrange the order of the variables
#car = car.iloc[:, [1, 0, 2, 3, 4]]
car.columns
car = pd.get_dummies(car)
#car.drop(['State'],axis =1,inplace=True)
# Correlation matrix 
a = car.corr()
a
#car.rename(columns={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms','Profit':'p'}, inplace=True)
# EDA
a1 = car.describe()

# Sctter plot and histogram between variables
sns.pairplot(car) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + Mfg_Month + KM + Automatic", data = car).fit()
model_train.summary()

# Prediction
pred = model_train.predict(car)
# Error
resid  = pred - car.Price
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car.iloc[:, 1:], car.Price)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car.iloc[:, 1:])

# Adjusted r-square
lasso.score(car.iloc[:, 1:], car.Price)

# RMSE
np.sqrt(np.mean((pred_lasso - car.Price)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car.iloc[:, 1:], car.Price)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car.columns[1:]))

rm.alpha

pred_rm = rm.predict(car.iloc[:, 1:])

# Adjusted r-square
rm.score(car.iloc[:, 1:], car.Price)

# RMSE
np.sqrt(np.mean((pred_rm - car.Price)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car.iloc[:, 1:], car.Price)

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car.columns[1:]))

enet.alpha

pred_enet = enet.predict(car.iloc[:, 1:])

# Adjusted r-square
enet.score(car.iloc[:, 1:], car.Price)

# RMSE
np.sqrt(np.mean((pred_enet - car.Price)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car.iloc[:, 1:], car.Price)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(car.iloc[:, 1:], car.Price)

# RMSE
np.sqrt(np.mean((lasso_pred - car.Price)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car.iloc[:, 1:], car.Price)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(car.iloc[:, 1:], car.Price)

# RMSE
np.sqrt(np.mean((ridge_pred - car.Price)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car.iloc[:, 1:], car.Price)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car.iloc[:, 1:], car.Price)

# RMSE
np.sqrt(np.mean((enet_pred - car.Price)**2))
######################################################################################
# Multilinear Regression with Regularization using L1 and L2 norm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import seaborn as sns

# loading the data
car = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/lasso ridge/Life_expectencey_LR.csv",encoding='latin1')
car.drop(['Country','Year','Status'],axis =1,inplace=True)
# Rearrange the order of the variables
#car = car.iloc[:, [1, 0, 2, 3, 4]]
car.columns
car = pd.get_dummies(car)
#car.drop(['State'],axis =1,inplace=True)
# Correlation matrix 
a = car.corr()
a
car = car.dropna()
#car.rename(columns={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms','Profit':'p'}, inplace=True)
# EDA
a1 = car.describe()

# Sctter plot and histogram between variables
sns.pairplot(car) # sp-hp, wt-vol multicolinearity issue

# Preparing the model on train data 
model_train = smf.ols("Life_expectancy ~ Adult_Mortality + infant_deaths + Alcohol + percentage_expenditure + Hepatitis_B + Measles + BMI + under_five_deaths + Polio + Total_expenditure + Diphtheria + HIV_AIDS + GDP + Population + thinness + thinness_yr + Income_composition + Schooling", data = car).fit() 
model_train.summary()

# Prediction
pred = model_train.predict(car)
# Error
resid  = pred - car.Life_expectancy
# RMSE value for data 
rmse = np.sqrt(np.mean(resid * resid))
rmse

# To overcome the issues, LASSO and RIDGE regression are used
################
###LASSO MODEL###
from sklearn.linear_model import Lasso
help(Lasso)

lasso = Lasso(alpha = 0.13, normalize = True)

lasso.fit(car.iloc[:, 1:], car.Life_expectancy)

# Coefficient values for all independent variables#
lasso.coef_
lasso.intercept_

plt.bar(height = pd.Series(lasso.coef_), x = pd.Series(car.columns[1:]))

lasso.alpha

pred_lasso = lasso.predict(car.iloc[:, 1:])

# Adjusted r-square
lasso.score(car.iloc[:, 1:], car.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_lasso - car.Life_expectancy)**2))


### RIDGE REGRESSION ###
from sklearn.linear_model import Ridge
help(Ridge)
rm = Ridge(alpha = 0.4, normalize = True)

rm.fit(car.iloc[:, 1:], car.Life_expectancy)

# Coefficients values for all the independent vairbales
rm.coef_
rm.intercept_

plt.bar(height = pd.Series(rm.coef_), x = pd.Series(car.columns[1:]))

rm.alpha

pred_rm = rm.predict(car.iloc[:, 1:])

# Adjusted r-square
rm.score(car.iloc[:, 1:], car.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_rm - car.Life_expectancy)**2))


### ELASTIC NET REGRESSION ###
from sklearn.linear_model import ElasticNet 
help(ElasticNet)
enet = ElasticNet(alpha = 0.4)

enet.fit(car.iloc[:, 1:], car.Life_expectancy)

# Coefficients values for all the independent vairbales
enet.coef_
enet.intercept_

plt.bar(height = pd.Series(enet.coef_), x = pd.Series(car.columns[1:]))

enet.alpha

pred_enet = enet.predict(car.iloc[:, 1:])

# Adjusted r-square
enet.score(car.iloc[:, 1:], car.Life_expectancy)

# RMSE
np.sqrt(np.mean((pred_enet - car.Life_expectancy)**2))


####################

# Lasso Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso

lasso = Lasso()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

lasso_reg = GridSearchCV(lasso, parameters, scoring = 'r2', cv = 5)
lasso_reg.fit(car.iloc[:, 1:], car.Life_expectancy)


lasso_reg.best_params_
lasso_reg.best_score_

lasso_pred = lasso_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
lasso_reg.score(car.iloc[:, 1:], car.Life_expectancy)

# RMSE
np.sqrt(np.mean((lasso_pred - car.Life_expectancy)**2))



# Ridge Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

ridge = Ridge()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

ridge_reg = GridSearchCV(ridge, parameters, scoring = 'r2', cv = 5)
ridge_reg.fit(car.iloc[:, 1:], car.Life_expectancy)

ridge_reg.best_params_
ridge_reg.best_score_

ridge_pred = ridge_reg.predict(car.iloc[:, 1:])

# Adjusted r-square#
ridge_reg.score(car.iloc[:, 1:], car.Life_expectancy)

# RMSE
np.sqrt(np.mean((ridge_pred - car.Life_expectancy)**2))



# ElasticNet Regression
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

enet = ElasticNet()

parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-4, 1e-3, 1e-2, 1, 5 ,10, 20]}

enet_reg = GridSearchCV(enet, parameters, scoring = 'neg_mean_squared_error', cv = 5)
enet_reg.fit(car.iloc[:, 1:], car.Life_expectancy)

enet_reg.best_params_
enet_reg.best_score_

enet_pred = enet_reg.predict(car.iloc[:, 1:])

# Adjusted r-square
enet_reg.score(car.iloc[:, 1:], car.Life_expectancy)

# RMSE
np.sqrt(np.mean((enet_pred - car.Life_expectancy)**2))
