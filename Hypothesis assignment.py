# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:13:19 2021

@author: Heeba
"""


%matplotlib inline
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest
import scipy
from scipy import stats


prom = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hypothesis/Cutlets.csv")
prom

prom.columns = "UnitA", "UnitB"
prom.dropna(inplace=True)

############ vis ###########
Unit_A=prom['UnitA'].mean()
Unit_B=prom['UnitB'].mean()

print('UnitA Mean = ',Unit_A, '\nUnitB Mean = ',Unit_B)
print('UnitA Mean > UnitB Mean = ',Unit_A>Unit_B)

sns.distplot(prom['UnitA'])
sns.distplot(prom['UnitB'])
plt.legend(['UnitA','UnitB'])
#############2-test#################
# Normality Test
stats.shapiro(prom.UnitA) # Shapiro Test

print(stats.shapiro(prom.UnitB))
help(stats.shapiro)

# Variance test
scipy.stats.levene(prom.UnitA, prom.UnitB)
help(scipy.stats.levene)
# p-value = 0.47 > 0.05 so p high null fly => Equal variances

# 2 Sample T test
scipy.stats.ttest_ind(prom.UnitA, prom.UnitB)
help(scipy.stats.ttest_ind)

############# One - Way Anova ################

# One - Way Anova
F, p = stats.f_oneway(prom.UnitA, prom.UnitB)
# p value
p  # P High Null Fly

######### 2-proportion test ###########

from statsmodels.stats.proportion import proportions_ztest

tab1 = prom.UnitA.value_counts()
tab1
tab2 = prom.UnitB.value_counts()
tab2

# crosstable table
pd.crosstab(prom.UnitA, prom.UnitB)

count = np.array([58, 152])
nobs = np.array([480, 740])

stats, prom = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(prom) # Pvalue 0.000

stats, prom = proportions_ztest(count, nobs, alternative = 'larger')
print(prom)  # Pvalue 0.999  
################ Chi-Square Test ################
np.asarray(prom)
count = pd.crosstab(prom.UnitA, prom.UnitB)
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

alpha=0.05
UnitA=pd.DataFrame(prom['UnitA'])
UnitB=pd.DataFrame(prom['UnitB'])
print(UnitA,UnitB)
Chi_square = tStat,pValue =sp.stats.ttest_ind(UnitA,UnitB)
Chi_square
############################################################################################3
LabTAT =pd.read_csv('C:/Users/DELL/Desktop/360 DIGITMG/hypothesis/lab_tat_updated.csv')
LabTAT.head()
LabTAT.describe()
Laboratory_1=LabTAT['Laboratory_1'].mean()
Laboratory_2=LabTAT['Laboratory_2'].mean()
Laboratory_3=LabTAT['Laboratory_3'].mean()
Laboratory_4=LabTAT['Laboratory_4'].mean()

print('Laboratory_1 Mean = ',Laboratory_1)
print('Laboratory_2 Mean = ',Laboratory_2)
print('Laboratory_3 Mean = ',Laboratory_3)
print('Laboratory_4 Mean = ',Laboratory_4)
############ vis#
sns.distplot(LabTAT['Laboratory_1'])
sns.distplot(LabTAT['Laboratory_2'])
sns.distplot(LabTAT['Laboratory_3'])
sns.distplot(LabTAT['Laboratory_4'])
plt.legend(['Laboratory_1','Laboratory_2','Laboratory_3','Laboratory_4'])

#############2-test#################
# Normality Test.
LabTAT.columns = "lab1", "lab2", "lab3", "lab4"
LabTAT.dropna(inplace=True)
np.asarray(LabTAT).astype(np.float64)

from scipy import stats
lab1 = stats.shapiro(LabTAT.lab1)
lab2 = stats.shapiro(LabTAT.lab2) # Shapiro Test
lab3 = stats.shapiro(LabTAT.lab3) # Shapiro Test
lab4 = stats.shapiro(LabTAT.lab4)
print(lab1,lab1,lab3,lab4)

scipy.stats.levene(LabTAT.lab1, LabTAT.lab2, LabTAT.lab3, LabTAT.lab4)

# 2 Sample T test
scipy.stats.ttest_ind(LabTAT.lab1, LabTAT.lab2, LabTAT.lab3, LabTAT.lab4)

############# One - Way Anova ################

# One - Way Anova
F, p = stats.f_oneway(LabTAT.lab1, LabTAT.lab2, LabTAT.lab3, LabTAT.lab4)
# p value
p  # P High Null Fly
######### 2-proportion test ###########
from statsmodels.stats.proportion import proportions_ztest

tab1 = LabTAT.lab1.value_counts()
tab1
tab2 = LabTAT.lab2.value_counts()
tab2
tab3 = LabTAT.lab3.value_counts()
tab3
tab4 = LabTAT.lab4.value_counts()
tab4
# crosstable table
pd.crosstab(LabTAT.lab1, LabTAT.lab2, LabTAT.lab3, LabTAT.lab4)

count = np.array([58, 152])
nobs = np.array([480, 740])

stats, LabTAT  = proportions_ztest(count, nobs, alternative = 'two-sided') 
print(LabTAT) # Pvalue 0.000

stats, LabTAT = proportions_ztest(count, nobs, alternative = 'larger')
print(LabTAT)  # Pvalue 0.999  
################ Chi-Square Test ################
np.asarray(LabTAT)
count = pd.crosstab(LabTAT.lab1, LabTAT.lab2, LabTAT.lab3, LabTAT.lab4)
count
Chisquares_results = scipy.stats.chi2_contingency(count)

Chi_square = [['Test Statistic', 'p-value'], [Chisquares_results[0], Chisquares_results[1]]]
Chi_square

Chi_square = tStat, pvalue = sp.stats.f_oneway(lab1,lab2,lab3,lab4)
Chi_square

####################################################

BR =pd.read_csv('C:/Users/DELL/Desktop/360 DIGITMG/hypothesis/BuyerRatio.csv')
BR.head()
BR.describe()
BR.drop('Observed Values',
  axis='columns', inplace=True)
East=BR['East'].mean()
West=BR['West'].mean()
North=BR['North'].mean()
South=BR['South'].mean()

print('East Mean = ',East)
print('West Mean = ',West)
print('North Mean = ',North)
print('South Mean = ',South)


#####vis
sns.distplot(BR['East'])
sns.distplot(BR['West'])
sns.distplot(BR['North'])
sns.distplot(BR['South'])
plt.legend(['East','West','North','South'])

alpha=0.05
Male = [50,142,131,70]
Female=[435,1523,1356,750]
Sales=[Male,Female]
print(Sales)
np.asarray(BR).astype(np.float64)

chiStats = sp.stats.chi2_contingency(Sales)
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')
if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')
  
  #critical value = 0.1
alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])# Find the critical value for 95% confidence*
                      #degree of freedom

observed_chi_val = chiStats[0]
#if observed chi-square < critical chi-square, then variables are not related
#if observed chi-square > critical chi-square, then variables are not independent (and hence may be related).
print('Interpret by critical value')
if observed_chi_val <= critical_value:
    # observed value is not in critical area therefore we accept null hypothesis
    print ('Null hypothesis cannot be rejected (variables are not related)')
else:
    # observed value is in critical area therefore we reject null hypothesis
    print ('Null hypothesis cannot be excepted (variables are not independent)')
####################################################################################3
Customer = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hypothesis/CustomerOrderform.csv")
Customer.head()
Customer.describe()

Customer.dropna(inplace=True)

Phillippines_value=Customer['Phillippines'].value_counts()
Indonesia_value=Customer['Indonesia'].value_counts()
Malta_value=Customer['Malta'].value_counts()
India_value=Customer['India'].value_counts()
print(Phillippines_value)
print(Indonesia_value)
print(Malta_value)
print(India_value)

chiStats = sp.stats.chi2_contingency([[271,267,269,280],[29,33,31,20]])
print('Test t=%f p-value=%f' % (chiStats[0], chiStats[1]))
print('Interpret by p-Value')
if chiStats[1] < 0.05:
  print('we reject null hypothesis')
else:
  print('we accept null hypothesis')
  
#critical value = 0.1
alpha = 0.05
critical_value = sp.stats.chi2.ppf(q = 1 - alpha,df=chiStats[2])
observed_chi_val = chiStats[0]
print('Interpret by critical value')
if observed_chi_val <= critical_value:
       print ('Null hypothesis cannot be rejected (variables are not related)')
else:
       print ('Null hypothesis cannot be excepted (variables are not independent)')
##############################################################################################
       
Fantaloons = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hypothesis/Fantaloons.csv")
Fantaloons.head()
Fantaloons.describe()

Fantaloons.dropna(inplace=True)
#vis
plt.hist(Fantaloons.Weekdays)
plt.hist(Fantaloons.Weekend)

Weekdays_value=Fantaloons['Weekdays'].value_counts()
Weekend_value=Fantaloons['Weekend'].value_counts()
print(Weekdays_value,Weekend_value)

tab = Fantaloons.groupby(['Weekdays', 'Weekend']).size()
count = np.array([280, 520]) #How many Male and Female
nobs = np.array([400, 400]) #Total number of Male and Female are there 

stat, pval = proportions_ztest(count, nobs,alternative='two-sided') 
#Alternative The alternative hypothesis can be either two-sided or one of the one- sided tests
#smaller means that the alternative hypothesis is prop < value
#larger means prop > value.
print('{0:0.3f}'.format(pval))
# two. sided -> means checking for equal proportions of Male and Female 
# p-value < 0.05 accept alternate hypothesis i.e.
# Unequal proportions 

stat, pval = proportions_ztest(count, nobs,alternative='larger')
print('{0:0.3f}'.format(pval))
# Ha -> Proportions of Male > Proportions of Female
# Ho -> Proportions of Female > Proportions of Male
# p-value >0.05 accept null hypothesis 
# so proportion of Female > proportion of Male