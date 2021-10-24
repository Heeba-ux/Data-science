# -*- coding: utf-8 -*-
"""
Created on Mon Jun 21 13:27:38 2021

@author: Heeba
"""


import lifelines

import pandas as pd
# Loading the the survival un-employment data
survival_unemp = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/survival analytics/Patient.csv")
survival_unemp.head()
survival_unemp.describe()

survival_unemp["Followup"].describe()

# Spell is referring to time 
T = survival_unemp.Followup

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_unemp.Eventtype)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_unemp.Scenario.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.Scenario==1], survival_unemp.Eventtype[survival_unemp.Scenario==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.Scenario==0], survival_unemp.Eventtype[survival_unemp.Scenario==0], label='0')
kmf.plot(ax=ax)
###################################################################################
import lifelines

import pandas as pd
# Loading the the survival un-employment data
survival_unemp = pd.read_excel("C:/Users/DELL/Desktop/360 DIGITMG/survival analytics/ECG_Surv.xlsx")
survival_unemp.head()
survival_unemp.describe()
survival_unemp = survival_unemp.drop(["name"], axis=1)
survival_unemp["survival_time_hr"].describe()


survival_unemp.dtypes


# Spell is referring to time 
T = survival_unemp.survival_time_hr

# Importing the KaplanMeierFitter model to fit the survival analysis
from lifelines import KaplanMeierFitter

# Initiating the KaplanMeierFitter model
kmf = KaplanMeierFitter()

# Fitting KaplanMeierFitter model on Time and Events for death 
kmf.fit(T, event_observed=survival_unemp.alive)

# Time-line estimations plot 
kmf.plot()

# Over Multiple groups 
# For each group, here group is ui
survival_unemp.group.value_counts()

# Applying KaplanMeierFitter model on Time and Events for the group "1"
kmf.fit(T[survival_unemp.group==1], survival_unemp.alive[survival_unemp.group==1], label='1')
ax = kmf.plot()

# Applying KaplanMeierFitter model on Time and Events for the group "0"
kmf.fit(T[survival_unemp.group==0], survival_unemp.alive[survival_unemp.group==0], label='0')
kmf.plot(ax=ax)
