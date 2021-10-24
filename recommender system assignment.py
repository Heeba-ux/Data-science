# -*- coding: utf-8 -*-
"""
Created on Thu May  6 10:52:47 2021

@author: Heeba
"""


import pandas as pd
import numpy as np
# import Dataset 
game = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/game.csv", encoding = 'utf8')
game.shape # shape
game.columns
game.game #
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(game, test_size = 0.2)
game = df_test
from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")
game["game"].isnull().sum() 
game["game"] = game["game"].fillna(" ")
# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(game.game)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
game_index = pd.Series(game.index, index = game['game']).drop_duplicates()
game_id = game_index["Rhythm Heaven"]
game_id
def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    game_id = game_index[game]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[game_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    game_idx  =  [i[0] for i in cosine_scores_N]
    game_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    game_similar_show = pd.DataFrame(columns=["game", "Score"])
    game_similar_show["game"] = game.loc[game_idx, "name"]
    game_similar_show["Score"] = game_scores
    game_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (game_similar_show)
    # return (anime_similar_show)
    
get_recommendations("Freekstyle", topN = 5)
game_index["Freekstyle"]

###############################################################################
###############################entertainment###################################

import pandas as pd
import numpy as np
# import Dataset 
movies = pd.read_csv("C:/Users/DELL/Desktop/360 DIGITMG/hands-on DataSets/entertainment.csv", encoding = 'utf8')
movies.shape # shape
movies.columns
movies.Titles #
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(game, test_size = 0.2)
movies = df_test
from sklearn.feature_extraction.text import TfidfVectorizer #term frequencey- inverse document frequncy is a numerical statistic that is intended to reflect how important a word is to document in a collecion or corpus
# Creating a Tfidf Vectorizer to remove all stop words
tfidf = TfidfVectorizer(stop_words = "english")
movies["Titles"].isnull().sum() 
movies["Titles"] = game["Titles"].fillna("")
# Preparing the Tfidf matrix by fitting and transforming
tfidf_matrix = tfidf.fit_transform(movies.Titles)   #Transform a count matrix to a normalized tf or tf-idf representation
tfidf_matrix.shape
from sklearn.metrics.pairwise import linear_kernel
cosine_sim_matrix = linear_kernel(tfidf_matrix, tfidf_matrix)
movies_index = pd.Series(movies.index, index = movies['Titles']).drop_duplicates()
movies_id = movies_index["Toy Story (1995)"]
movies_id
def get_recommendations(Name, topN):    
    # topN = 10
    # Getting the movie index using its title 
    movies_id = movies_index[Titles]
    
    # Getting the pair wise similarity score for all the anime's with that 
    # anime
    cosine_scores = list(enumerate(cosine_sim_matrix[movies_id]))
    
    # Sorting the cosine_similarity scores based on scores 
    cosine_scores = sorted(cosine_scores, key=lambda x:x[1], reverse = True)
    
    # Get the scores of top N most similar movies 
    cosine_scores_N = cosine_scores[0: topN+1]
    
    # Getting the movie index 
    movies_idx  =  [i[0] for i in cosine_scores_N]
    movies_scores =  [i[1] for i in cosine_scores_N]
    
    # Similar movies and scores
    movies_similar_show = pd.DataFrame(columns=["movies", "Reviews"])
    movies_similar_show["Titles"] = movies.loc[movies_idx, "Titles"]
    movies_similar_show["Reviews"] = movies_scores
    movies_similar_show.reset_index(inplace = True)  
    # anime_similar_show.drop(["index"], axis=1, inplace=True)
    print (movies_similar_show)
    # return (anime_similar_show)
    
get_recommendations("Toy Story (1995)", topN = 10)
movies_index["Toy Story (1995)"]
