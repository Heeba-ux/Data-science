# -*- coding: utf-8 -*-
"""
Created on Wed May 26 10:25:29 2021

@author: heeba
"""
###################### calories data ####################################3

# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

wcat = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/simple LR/calories_consumed.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

wcat.describe()
wcat.rename(columns={'Weight gained (grams)':'AT','Calories Consumed':'Waist'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.bar(height = wcat.AT, x = np.arange(1, 110, 1))
plt.hist(wcat.AT) #histogram
plt.boxplot(wcat.AT) #boxplot

plt.bar(height = wcat.Waist, x = np.arange(1, 110, 1))
plt.hist(wcat.Waist) #histogram
plt.boxplot(wcat.Waist) #boxplot

# Scatter plot
plt.scatter(x = wcat['Waist'], y = wcat['AT'], color = 'green') 

# correlation
np.corrcoef(wcat.Waist, wcat.AT) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(wcat.Waist, wcat.AT)[0, 1]
cov_output

# wcat.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('AT ~ Waist', data = wcat).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(wcat['Waist']))

# Regression Line
plt.scatter(wcat.Waist, wcat.AT)
plt.plot(wcat.Waist, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = wcat.AT - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(wcat['Waist']), y = wcat['AT'], color = 'brown')
np.corrcoef(np.log(wcat.Waist), wcat.AT) #correlation

model2 = smf.ols('AT ~ np.log(Waist)', data = wcat).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(wcat['Waist']))

# Regression Line
plt.scatter(np.log(wcat.Waist), wcat.AT)
plt.plot(np.log(wcat.Waist), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = wcat.AT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = wcat['Waist'], y = np.log(wcat['AT']), color = 'orange')
np.corrcoef(wcat.Waist, np.log(wcat.AT)) #correlation

model3 = smf.ols('np.log(AT) ~ Waist', data = wcat).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = wcat.AT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)

model4 = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = wcat).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(wcat))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = wcat.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = wcat.AT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(wcat, test_size = 0.2)

finalmodel = smf.ols('np.log(AT) ~ Waist + I(Waist*Waist)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Waist = np.exp(test_pred)
pred_test_Waist

# Model Evaluation on Test data
test_res = test.Waist - pred_test_Waist
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Waist = np.exp(train_pred)
pred_train_Waist

# Model Evaluation on train data
train_res = train.Waist - pred_train_Waist
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
##############################################################################
################################### Delivary time ############################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

dtst = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/simple LR/delivery_time.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

dtst.describe()
dtst.rename(columns={'Delivery Time':'DT','Sorting Time':'ST'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.hist(dtst.DT) #histogram
plt.boxplot(dtst.DT) #boxplot


plt.hist(dtst.ST) #histogram
plt.boxplot(dtst.ST) #boxplot

# Scatter plot
plt.scatter(x = dtst['ST'], y = dtst['DT'], color = 'green') 

# correlation
np.corrcoef(dtst.ST, dtst.DT) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(dtst.ST, dtst.DT)[0, 1]
cov_output

# dtst.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('DT ~ ST', data = dtst).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(dtst['ST']))

# Regression Line
plt.scatter(dtst.ST, dtst.DT)
plt.plot(dtst.ST, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = dtst.DT - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(dtst['ST']), y = dtst['DT'], color = 'brown')
np.corrcoef(np.log(dtst.ST), dtst.DT) #correlation

model2 = smf.ols('DT ~ np.log(ST)', data = dtst).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(dtst['ST']))

# Regression Line
plt.scatter(np.log(dtst.ST), dtst.DT)
plt.plot(np.log(dtst.ST), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = dtst.DT - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = dtst['ST'], y = np.log(dtst['DT']), color = 'orange')
np.corrcoef(dtst.ST, np.log(dtst.DT)) #correlation

model3 = smf.ols('np.log(DT) ~ ST', data = dtst).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(dtst['ST']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(dtst.ST, np.log(dtst.DT))
plt.plot(dtst.DT, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = dtst.DT - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
plt.scatter(x = dtst['ST'], y = np.log(dtst['DT']), color = 'pink')
np.corrcoef(dtst.ST, np.log(dtst.DT)) 

model4 = smf.ols('np.log(DT) ~ ST + I(ST*ST)', data = dtst).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(dtst))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = dtst.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(dtst.ST, np.log(dtst.DT))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = dtst.DT - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(dtst, test_size = 0.2)

finalmodel = smf.ols('np.log(DT) ~ ST + I(ST*ST)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_ST = np.exp(test_pred)
pred_test_ST

# Model Evaluation on Test data
test_res = test.ST - pred_test_ST
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_ST = np.exp(train_pred)
pred_train_ST

# Model Evaluation on train data
train_res = train.ST - pred_train_ST
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
#############################################################################
############################## emp data #####################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

ed = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/simple LR/emp_data.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

ed.describe()
ed.rename(columns={'Salary_hike':'sh','Churn_out_rate':'cr'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.hist(ed.sh) #histogram
plt.boxplot(ed.sh) #boxplot


plt.hist(ed.cr) #histogram
plt.boxplot(ed.cr) #boxplot

# Scatter plot
plt.scatter(x = ed['sh'], y = ed['cr'], color = 'green') 

# correlation
np.corrcoef(ed.sh, ed.cr) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(ed.sh, ed.cr)[0, 1]
cov_output

# dtst.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('sh ~ cr', data = ed).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(ed['cr']))

# Regression Line
plt.scatter(ed.sh, ed.cr)
plt.plot(ed.sh, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = ed.sh - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(ed['cr']), y = ed['sh'], color = 'brown')
np.corrcoef(np.log(ed.cr), ed.sh) #correlation

model2 = smf.ols('sh ~ np.log(cr)', data = ed).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(ed['cr']))

# Regression Line
plt.scatter(np.log(ed.cr), ed.sh)
plt.plot(np.log(ed.sh), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = ed.sh - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = ed['cr'], y = np.log(ed['sh']), color = 'orange')
np.corrcoef(ed.cr, np.log(ed.sh)) #correlation

model3 = smf.ols('np.log(sh) ~ cr', data = ed).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(ed['cr']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(ed.cr, np.log(ed.sh))
plt.plot(ed.sh, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = ed.sh - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
plt.scatter(x = ed['cr'], y = np.log(ed['sh']), color = 'pink')
np.corrcoef(ed.cr, np.log(ed.sh)) 

model4 = smf.ols('np.log(sh) ~ cr + I(cr*cr)', data = ed).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(ed))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = ed.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(ed.cr, np.log(ed.sh))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = ed.sh - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(ed, test_size = 0.2)

finalmodel = smf.ols('np.log(sh) ~ cr + I(cr*cr)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_cr = np.exp(test_pred)
pred_test_cr

# Model Evaluation on Test data
test_res = test.cr - pred_test_cr
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_cr = np.exp(train_pred)
pred_train_cr

# Model Evaluation on train data
train_res = train.cr - pred_train_cr
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
############################################################################
############################ salary ########################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

ed = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/simple LR/Salary_Data.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

ed.describe()
ed.rename(columns={'YearsExperience':'ye','Salary':'sl'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.hist(ed.ye) #histogram
plt.boxplot(ed.ye) #boxplot


plt.hist(ed.sl) #histogram
plt.boxplot(ed.sl) #boxplot

# Scatter plot
plt.scatter(x = ed['ye'], y = ed['sl'], color = 'green') 

# correlation
np.corrcoef(ed.ye, ed.sl) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(ed.ye, ed.sl)[0, 1]
cov_output

# dtst.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('ye ~ sl', data = ed).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(ed['sl']))

# Regression Line
plt.scatter(ed.ye, ed.sl)
plt.plot(ed.ye, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = ed.ye - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(ed['sl']), y = ed['ye'], color = 'brown')
np.corrcoef(np.log(ed.sl), ed.ye) #correlation

model2 = smf.ols('ye ~ np.log(sl)', data = ed).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(ed['sl']))

# Regression Line
plt.scatter(np.log(ed.sl), ed.ye)
plt.plot(np.log(ed.ye), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = ed.ye - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = ed['sl'], y = np.log(ed['ye']), color = 'orange')
np.corrcoef(ed.sl, np.log(ed.ye)) #correlation

model3 = smf.ols('np.log(ye) ~ sl', data = ed).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(ed['sl']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(ed.sl, np.log(ed.ye))
plt.plot(ed.ye, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = ed.ye - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
plt.scatter(x = ed['sl'], y = np.log(ed['ye']), color = 'pink')
np.corrcoef(ed.sl, np.log(ed.ye)) 

model4 = smf.ols('np.log(ye) ~ sl + I(sl*sl)', data = ed).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(ed))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = ed.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(ed.sl, np.log(ed.ye))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = ed.ye - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(ed, test_size = 0.2)

finalmodel = smf.ols('np.log(ye) ~ sl + I(sl*sl)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_sl = np.exp(test_pred)
pred_test_sl

# Model Evaluation on Test data
test_res = test.sl - pred_test_sl
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_sl = np.exp(train_pred)
pred_train_sl

# Model Evaluation on train data
train_res = train.sl - pred_train_sl
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
#######################################################################
################### sat score ##########################################
# Importing necessary libraries
import pandas as pd # deals with data frame  
import numpy as np  # deals with numerical values

ed = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/simple LR/SAT_GPA.csv")

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

ed.describe()
ed.rename(columns={'SAT_Scores':'sh','GPA':'cr'}, inplace=True)
#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

plt.hist(ed.sh) #histogram
plt.boxplot(ed.sh) #boxplot


plt.hist(ed.cr) #histogram
plt.boxplot(ed.cr) #boxplot

# Scatter plot
plt.scatter(x = ed['sh'], y = ed['cr'], color = 'green') 

# correlation
np.corrcoef(ed.sh, ed.cr) 

# Covariance
# NumPy does not have a function to calculate the covariance between two variables directly. 
# Function for calculating a covariance matrix called cov() 
# By default, the cov() function will calculate the unbiased or sample covariance between the provided random variables.

cov_output = np.cov(ed.sh, ed.cr)[0, 1]
cov_output

# dtst.cov()


# Import library
import statsmodels.formula.api as smf

# Simple Linear Regression
model = smf.ols('sh ~ cr', data = ed).fit()
model.summary()

pred1 = model.predict(pd.DataFrame(ed['cr']))

# Regression Line
plt.scatter(ed.sh, ed.cr)
plt.plot(ed.sh, pred1, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res1 = ed.sh - pred1
res_sqr1 = res1 * res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1

######### Model building on Transformed Data
# Log Transformation
# x = log(waist); y = at

plt.scatter(x = np.log(ed['cr']), y = ed['sh'], color = 'brown')
np.corrcoef(np.log(ed.cr), ed.sh) #correlation

model2 = smf.ols('sh ~ np.log(cr)', data = ed).fit()
model2.summary()

pred2 = model2.predict(pd.DataFrame(ed['cr']))

# Regression Line
plt.scatter(np.log(ed.cr), ed.sh)
plt.plot(np.log(ed.sh), pred2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res2 = ed.sh - pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2


#### Exponential transformation
# x = waist; y = log(at)

plt.scatter(x = ed['cr'], y = np.log(ed['sh']), color = 'orange')
np.corrcoef(ed.cr, np.log(ed.sh)) #correlation

model3 = smf.ols('np.log(sh) ~ cr', data = ed).fit()
model3.summary()

pred3 = model3.predict(pd.DataFrame(ed['cr']))
pred3_at = np.exp(pred3)
pred3_at

# Regression Line
plt.scatter(ed.cr, np.log(ed.sh))
plt.plot(ed.sh, pred3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Error calculation
res3 = ed.sh - pred3_at
res_sqr3 = res3 * res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3


#### Polynomial transformation
# x = waist; x^2 = waist*waist; y = log(at)
plt.scatter(x = ed['cr'], y = np.log(ed['sh']), color = 'pink')
np.corrcoef(ed.cr, np.log(ed.sh)) 

model4 = smf.ols('np.log(sh) ~ cr + I(cr*cr)', data = ed).fit()
model4.summary()

pred4 = model4.predict(pd.DataFrame(ed))
pred4_at = np.exp(pred4)
pred4_at

# Regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = ed.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)
# y = wcat.iloc[:, 1].values


plt.scatter(ed.cr, np.log(ed.sh))
plt.plot(X, pred4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()


# Error calculation
res4 = ed.sh - pred4_at
res_sqr4 = res4 * res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4


# Choose the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([rmse1, rmse2, rmse3, rmse4])}
table_rmse = pd.DataFrame(data)
table_rmse

###################
# The best model

from sklearn.model_selection import train_test_split

train, test = train_test_split(ed, test_size = 0.2)

finalmodel = smf.ols('np.log(sh) ~ cr + I(cr*cr)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_cr = np.exp(test_pred)
pred_test_cr

# Model Evaluation on Test data
test_res = test.cr - pred_test_cr
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse


# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_cr = np.exp(train_pred)
pred_train_cr

# Model Evaluation on train data
train_res = train.cr - pred_train_cr
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse
