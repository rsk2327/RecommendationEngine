# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:00:34 2016

@author: rsk
"""

import os
os.chdir("/home/rsk/Documents/RecommenderProject/RecommendationEngine/python")
import pandas as pd
import numpy as np
import datetime
from dateutil import parser
import dateutil
from sklearn.cross_validation import train_test_split
from utils import *
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import pylab
from compALS import *
from collections import Counter
from sklearn.neighbors import NearestNeighbors


print('Importing data.')
ratingData,movieData,userData = importData("/home/rsk/Documents/RecommenderProject/")

#%%


data = pd.merge(ratingData,userData,how="left",on="userID")
data = pd.merge(data,movieData,how="left",on="movieID")
data = cleanData(data)  #removes unneccesary rows and columns
data = dateTime(data)

#%%

data_sub = data[['userID','movieID','rating']]

pivot = data_sub.pivot_table('rating','userID','movieID')
pivot=pivot.fillna(0)
#%%

mask = (pivot.values>0.5).astype("int")

sparseData = csr_matrix(pivot.values)
als = nf.Lsnmf(sparseData,seed="random_vcol",rank=2,max_iter=30,beta=0.5)
als_fit= als.factorize()

user_features = np.array(als_fit.basis().todense())
movie_features = np.array(als_fit.coef().todense())

dataset = pd.DataFrame(user_features)

#%% ###################%% CLUSTERING #####################################


from sklearn.cluster import KMeans
num_clusters=2
clust = KMeans(n_clusters=num_clusters,max_iter=300,n_init=200,n_jobs=-1)
clust = clust.fit_predict(dataset)

dataset["cluster"] = (clust+1.0).astype("int")
dataset["ID"] = list(range(1,944))
dataset["gender"] = userData["gender"]
dataset["occupation"] = userData["occupation"]

#Plotting
from ggplot import *
x= np.array(dataset[0])
y= np.array(dataset[1])

p = ggplot(dataset, aes(x=0,y=1,z=2,color="cluster")) +geom_point() +ggtitle("User clusters")
print p





#%%  COLLABORATIVE RECOMMENDATION ######################### 


def getRecommendations_user(userID,user_features, neighborCount=10,neighborMovieCount=10,moviesToRecommend=10,returnCount = False,
                            returnWatched=True):
    """
    Uses collaborative filtering to make recommendations for a specific user. 
    The function finds the closest neighbors to the given user and then take their highest rated movies.
    A simple frequency count of the the movies is used to make recommendations
    
    INPUT
    
    userID : ID of the user for whom the recommendations are to be made
    
    user_features : Feature vectors for all users in the dataset
    
    neighborCount : Number of closest neighbors to be considered while making recommendations
    
    neighborMovieCount : Number of movies to be considered from each of the selected neighbors.
                         The top higheset rated movies by the neighbors are considered for recommendations
                         
    moviesToRecommend : Number of movies to recommend to the user
    
    returnCount : Whether to return the counts of the recommended movies
    
    returnWatched : Whether movies watched by the user should be recommended. If false, only movies not watched by the user
    are considered
    
    """
    
    neighborModel = NearestNeighbors()
    nearestNeighbors = neighborModel.fit(X=user_features)
    nearest= nearestNeighbors.kneighbors(user_features,n_neighbors=neighborCount,return_distance=False)[:,1:neighborCount]
    nearestNeighborList = nearest[userID]
    
    
    watchedMovies = data[data['userID']==userID][['movieID']].values
    
    movieList = []
    for i in range(len(nearestNeighborList)):
        user = nearestNeighborList[i]
        userRatingData = data[data['userID']==user][['userID','movieID','rating']].sort_values('rating',ascending=False)
        if returnWatched==False:
            userRatingData = userRatingData[ [not i in watchedMovies for i in userRatingData['movieID'].values] ]
        movieList+=list(userRatingData[:neighborMovieCount]['movieID'].values)
    
        
                
    movieList = Counter(movieList)
    
    recommendList = []
    movieCountList = movieList.most_common(moviesToRecommend)
    if returnCount==False:
        for i in range(moviesToRecommend):
            recommendList.append( movieData[movieData['movieID']==movieCountList[i][0]]['movie title'].values[0] )
    else:
        for i in range(moviesToRecommend):
            recommendList.append( movieData[movieData['movieID']==movieCountList[i][0]]['movie title'].values[0], movieCountList[i][1] )
        
    return recommendList
   

def getTopMovies(userID,count):
    
    return data[data['userID']==userID].sort_values('rating',ascending=False)['movie title'][:count].values
    
    
def compareResults(list1,list2):
    
    count = 0
    
    for i in range(len(list1)):
        if list1[i] in list2:
            count += 1
            
    return count



#%%
w= getRecommendations_user(1,user_features,25,55,50,False,returnWatched = False)
w2 = getTopMovies(1,50)
compareResults(w,w2)

#%%
w= getRecommendations_user(1,user_features,25,55,50,False,returnWatched =True)
w2 = getTopMovies(1,50)
compareResults(w,w2)

#%%
w= getRecommendations_user(3,user_features,25,10,50,False,returnWatched =True)
w2 = getTopMovies(3,50)
compareResults(w,w2)