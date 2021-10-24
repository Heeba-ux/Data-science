# -*- coding: utf-8 -*-
"""
Created on Tue May 25 13:06:16 2021

@author: DELL
"""


from scipy import stats
from scipy.stats import norm

# Z-score of 90% confidence interval 
stats.norm.ppf(0.95)

# Z-score of 94% confidence interval
stats.norm.ppf(0.97)

# Z-score of 60% confidence interval
stats.norm.ppf(0.8)

######################################################


import numpy as np
from scipy import stats
from scipy.stats import norm

Mean = 5+7
print('Mean Profit is Rs', Mean*45,'Million')

SD = (3^2)+(4^2)
print('Standard Deviation is Rs', SD*45, 'Million')

# A. Specify a Rupee range (centered on the mean) such that it contains 95% probability for the annual profit of the company.
print('Range is Rs',(stats.norm.interval(0.95,540,315)),'in Millions')

# B. Specify the 5th percentile of profit (in Rupees) for the company
# To compute 5th Percentile, we use the formula X=μ + Zσ; wherein from z table, 5 percentile = -1.64
X= 540+(-1.64)*(315)
print('5th percentile of profit (in Million Rupees) is',np.round(X,2))

# C. Which of the two divisions has a larger probability of making a loss in a given year?
# Probability of Division 1 making a loss P(X<0)
stats.norm.cdf(0,5,3)
# Probability of Division 2 making a loss P(X<0)
stats.norm.cdf(0,7,4)
