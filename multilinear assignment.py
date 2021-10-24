# -*- coding: utf-8 -*-
"""
Created on Thu May 27 10:30:19 2021

@author: heeba
"""


# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cars = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/multi lr/50_Startups.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

cars.describe()
cars.rename(columns={'R&D Spend':'rd','Administration':'ad','Marketing Spend':'ms','Profit':'p'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# HP
plt.bar(height = cars.p, x = np.arange(1, 82, 1))
plt.hist(cars.p) #histogram
plt.boxplot(cars.p) #boxplot

# MPG
plt.bar(height = cars.ms, x = np.arange(1, 82, 1))
plt.hist(cars.ms) #histogram
plt.boxplot(cars.ms) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cars['p'], y=cars['ms'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cars['p'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cars.ms, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cars.iloc[:, :])
                             
# Correlation matrix 
cars.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('ms ~ p + rd + State + ad', data = cars).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cars_new = cars.drop(cars.index[[40]])

# Preparing model                  
ml_new = smf.ols('ms ~ p + rd + State + ad', data = cars_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('rd ~ ad + ms + State', data = cars).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('ad ~ rd + ms + p', data = cars).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('ms ~ rd + p + State', data = cars).fit().rsquared  
vif_vol = 1/(1 - rsq_vol) 

rsq_sp = smf.ols('p ~ ms + rd + ad', data = cars).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['rd', 'ad', 'ms', 'State'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('p ~ ms + rd + State', data = cars).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cars)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cars.p, lowess = True)
plt.xlabel('profit')
plt.ylabel('marketing spend')
plt.title('profit vs marketing spend')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cars_train, cars_test = train_test_split(cars, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("p ~ ms + rd + State", data = cars_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cars_test)
test_pred
# test residual values 
test_resid = test_pred - cars_test.p
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cars_train)
train_pred
# train residual values 
train_resid  = train_pred - cars_train.p
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse

#######################################################################
# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cd = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/multi lr/Computer_Data.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
cd.columns
cd.describe()
cd = cd.drop(columns=['Unnamed: 0', 'cd', 'multi', 'premium'])

cars.rename(columns={'price':'p','speed':'s','Marketing Spend':'ms','Profit':'p'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = cd.price, x = np.arange(1, 82, 1))
plt.hist(cd.price) #histogram
plt.boxplot(cd.price) #boxplot

# speed
plt.bar(height = cd.speed, x = np.arange(1, 82, 1))
plt.hist(cd.speed) #histogram
plt.boxplot(cd.speed) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cd['price'], y=cd['speed'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cd['price'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cd.trend, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cd.iloc[:, :])
                             
# Correlation matrix 
cd.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ speed + hd + ram + trend', data = cd).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cd_new = cd.drop(cd.index[[40]])

# Preparing model                  
ml_new = smf.ols('price ~ speed + hd + ram + trend', data = cd_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('price ~ speed + hd + ram', data = cd).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('speed ~ price + hd + trend', data = cd).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('trend ~ speed + price + ram', data = cd).fit().rsquared  
vif_vol = 1/(1 - rsq_vol) 

rsq_sp = smf.ols('ram ~ price + speed + trend', data = cd).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['ram', 'hd', 'speed', 'trend'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ speed + ram + trend', data = cd).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cd)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cd.price, lowess = True)
plt.xlabel('price')
plt.ylabel('trend')
plt.title('price vs trend')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cd_train, cd_test = train_test_split(cd, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ speed + ram + trend", data = cd_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cd_test)
test_pred
# test residual values 
test_resid = test_pred - cd_test.price
test_resid
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cd_train)
train_pred
# train residual values 
train_resid  = train_pred - cd_train.price
train_resid
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
##################################################################
import pandas as pd
import numpy as np

# loading the data
cd = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/multi lr/ToyotaCorolla.csv",encoding='latin1')
cd = pd.get_dummies(cd)
# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
cd.columns
cd.describe()
#cd = cd.drop(columns=['Unnamed: 0', 'cd', 'multi', 'premium'])

#cars.rename(columns={'price':'p','speed':'s','Marketing Spend':'ms','Profit':'p'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = cd.price, x = np.arange(1, 82, 1))
plt.hist(cd.Price) #histogram
plt.boxplot(cd.Price) #boxplot

# speed
plt.bar(height = cd.speed, x = np.arange(1, 82, 1))
plt.hist(cd.Fuel_Type) #histogram
plt.boxplot(cd.Age_08_04) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cd['Price'], y=cd['Automatic'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cd['Price'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cd.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cd.iloc[:, :])
                             
# Correlation matrix 
cd.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + Mfg_Month + KM + Automatic', data = cd).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cd_new = cd.drop(cd.index[[100]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + Mfg_Month + KM + Automatic', data = cd_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('Price ~ cc + Gears + Weight', data = cd).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('Mfg_Year ~ Price + Gears + Tow_Bar', data = cd).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('Gears ~ Price + Quarterly_Tax + Doors', data = cd).fit().rsquared  
vif_vol = 1/(1 - rsq_vol) 

rsq_sp = smf.ols('Guarantee_Period ~ Gears + Automatic + Met_Color', data = cd).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Gears', 'Automatic', 'Met_Color', 'Mfg_Year'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04 + Mfg_Month + KM + Automatic', data = cd).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cd)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cd.Price, lowess = True)
plt.xlabel('price')
plt.ylabel('all')
plt.title('price vs all')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cd_train, cd_test = train_test_split(cd, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Price ~ Age_08_04 + Mfg_Month + KM + Automatic", data = cd_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cd_test)
test_pred
# test residual values 
test_resid = test_pred - cd_test.Price
test_resid
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse


# train_data prediction
train_pred = model_train.predict(cd_train)
train_pred
# train residual values 
train_resid  = train_pred - cd_train.Price
train_resid
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
#####################################################################
# Multilinear Regression
import pandas as pd
import numpy as np

# loading the data
cd = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/multi lr/Avacado_Price.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)
cd.columns
cd.describe()
cd = cd.drop(columns=['Unnamed: 0', 'cd', 'multi', 'premium'])

cd.rename(columns={'AveragePrice':'price','Total_Volume':'tv','tot_ava1':'ava1','tot_ava2':'ava2','tot_ava3':'ava3','Total_Bags':'bags','Small_Bags':'s','Large_Bags':'l','XLarge Bags':'xl'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# price
plt.bar(height = cd.price, x = np.arange(1, 82, 1))
plt.hist(cd.price) #histogram
plt.boxplot(cd.price) #boxplot

# speed
plt.bar(height = cd.speed, x = np.arange(1, 82, 1))
plt.hist(cd.region) #histogram
plt.boxplot(cd.xl) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=cd['price'], y=cd['tv'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(cd['price'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(cd.year, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(cd.iloc[:, :])
                             
# Correlation matrix 
cd.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('price ~ tv + ava1 + ava2 + ava3', data = cd).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

cd_new = cd.drop(cd.index[[40]])

# Preparing model                  
ml_new = smf.ols('price ~ tv + ava1 + ava2 + ava3', data = cd_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('price ~ bags + s + l', data = cd).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('bags ~ s + l + xl', data = cd).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('year ~ tv + price + bags', data = cd).fit().rsquared  
vif_vol = 1/(1 - rsq_vol) 

rsq_sp = smf.ols('ava1 ~ ava2 + ava3 + bags', data = cd).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['bags', 'l', 'xl', 's'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('price ~ tv + ava1 + ava2 + ava3', data = cd).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(cd)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = cd.price, lowess = True)
plt.xlabel('price')
plt.ylabel('type')
plt.title('price vs type')
plt.show()

sm.graphics.influence_plot(final_ml)


### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
cd_train, cd_test = train_test_split(cd, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("price ~ tv + ava1 + ava2 + ava3", data = cd_train).fit()

# prediction on test data set 
test_pred = model_train.predict(cd_test)
test_pred
# test residual values 
test_resid = test_pred - cd_test.price
test_resid
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(cd_train)
train_pred
# train residual values 
train_resid  = train_pred - cd_train.price
train_resid
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
