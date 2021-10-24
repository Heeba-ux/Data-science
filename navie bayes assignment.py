# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:02:06 2021

@author: Heeba
"""


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
slr_train = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/navie bayes/SalaryData_Train.csv",encoding = "ISO-8859-1")
slr_test = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/navie bayes/SalaryData_Test.csv",encoding = "ISO-8859-1")

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

#Bag of Words
slr_bow = CountVectorizer(analyzer = split_into_words).fit(slr_train.Salary)
slr_bow1 = CountVectorizer(analyzer = split_into_words).fit(slr_test.Salary)
# Defining BOW for all 
all_slr_train_matrix = slr_bow.transform(slr_train.Salary)
all_slr_test_matrix = slr_bow.transform(slr_test.Salary)
# For training messages
train_slr_train_matrix = slr_bow.transform(slr_train.Salary)

# For testing messages
test_slr_test_matrix = slr_bow1.transform(slr_test.Salary)

# Learning Term weighting and normalizing on entire salaries
tfidf_transformer = TfidfTransformer().fit(all_slr_train_matrix)
tfidf_transformer1 = TfidfTransformer().fit(all_slr_test_matrix)
# Preparing TFIDF for train salary
train_tfidf = tfidf_transformer.transform(train_slr_train_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test salary
test_tfidf = tfidf_transformer1.transform(test_slr_test_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, slr_train.Salary)

classifier_mb1 = MB()
classifier_mb1.fit(test_tfidf, slr_test.Salary)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == slr_test.Salary)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, slr_test.Salary) 

pd.crosstab(test_pred_m, slr_test.Salary)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == slr_train.Salary)
accuracy_train_m


classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, slr_train.Salary)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == slr_test.Salary)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, slr_test.Salary) 

pd.crosstab(test_pred_lap, slr_test.Salary)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == slr_train.Salary)
accuracy_train_lap

######################################################################
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
cars = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/navie bayes/NB_Car_Ad.csv",encoding = "ISO-8859-1")
cars.head()
#assigning independent and dependent variables
x = cars.iloc[:,2:-1].values
y = cars.iloc[:,-1].values

#splitting data in testing and training set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0)
#applying feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#training model
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

#getting confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = nb.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print('confusion matrix:\n',cm)

#checking accuracy
from sklearn.metrics import accuracy_score
nba = accuracy_score(y_test,y_pred)
print('accuracy score = ',accuracy_score(y_test,y_pred))

#######################################################################
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

# Loading the data set
twt_data = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/navie bayes/Disaster_tweets_NB.csv",encoding = "ISO-8859-1")

# cleaning data 
import re
stop_words = []
# Load the custom built Stopwords
with open("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/textmining/stopwords_en.txt","r") as sw:
    stop_words = sw.read()

stop_words = stop_words.split("\n")
   
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

# testing above function with sample text => removes punctuations, numbers
cleaning_text("Hope you are having a good week. Just checking in")
cleaning_text("hope i can understand your feelings 123121. 123 hi how .. are you?")
cleaning_text("Hi how are you, I am good")

twt_data.text = twt_data.text.apply(cleaning_text)

# removing empty rows
twt_data = twt_data.loc[twt_data.text != " ",:]

# CountVectorizer
# Convert a collection of text documents to a matrix of token counts

# splitting data into train and test data sets 
from sklearn.model_selection import train_test_split

twt_train, twt_test = train_test_split(txt_data, test_size = 0.2)

# creating a matrix of token counts for the entire text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Bag of Words
twts_bow = CountVectorizer(analyzer = split_into_words).fit(twt_data.text)

# Defining BOW for all messages
all_twts_matrix = twts_bow.transform(twt_data.text)

# For training messages
train_twts_matrix = twts_bow.transform(twt_train.text)

# For testing messages
test_twts_matrix = twts_bow.transform(twt_test.text)

# Learning Term weighting and normalizing on entire tweets
tfidf_transformer = TfidfTransformer().fit(all_twts_matrix)

# Preparing TFIDF for train 
train_tfidf = tfidf_transformer.transform(train_twts_matrix)
train_tfidf.shape # (row, column)

# Preparing TFIDF for test 
test_tfidf = tfidf_transformer.transform(test_twts_matrix)
test_tfidf.shape #  (row, column)

# Preparing a naive bayes model on training data set 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, twt_train.text)

# Evaluation on Test Data
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == twt_test.text)
accuracy_test_m

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_m, twt_test.text) 

pd.crosstab(test_pred_m, twt_test.text)

# Training Data accuracy
train_pred_m = classifier_mb.predict(train_tfidf)
accuracy_train_m = np.mean(train_pred_m == twt_train.text)
accuracy_train_m

classifier_mb_lap = MB(alpha = 3)
classifier_mb_lap.fit(train_tfidf, twt_train.text)

# Evaluation on Test Data after applying laplace
test_pred_lap = classifier_mb_lap.predict(test_tfidf)
accuracy_test_lap = np.mean(test_pred_lap == twt_test.text)
accuracy_test_lap

from sklearn.metrics import accuracy_score
accuracy_score(test_pred_lap, twt_test.text) 

pd.crosstab(test_pred_lap, twt_test.text)

# Training Data accuracy
train_pred_lap = classifier_mb_lap.predict(train_tfidf)
accuracy_train_lap = np.mean(train_pred_lap == twt_train.text)
accuracy_train_lap
