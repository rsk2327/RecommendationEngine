import os
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
ratingData,movieData,userData = importData("/home/satvik/Analytics/Recommender Project/")

neighborCount=5
data = pd.merge(ratingData,userData,how="left",on="userID")
data = pd.merge(data,movieData,how="left",on="movieID")
data = cleanData(data)  #removes unneccesary rows and columns
data = dateTime(data)

#%%

data_sub = data[['userID','movieID','rating']]

pivot = data_sub.pivot_table('rating','userID','movieID')
pivot=pivot.fillna(0)                             #pivot is matrix of ratings with users as rows as movies as columns
#%%

neighborModel = NearestNeighbors()
nearestNeighbors = neighborModel.fit(X=movieData.ix[:,6:-1])
nearest= nearestNeighbors.kneighbors(movieData.ix[:,6:-1],n_neighbors=neighborCount,return_distance=False)[:,0:neighborCount]


def getRecommendations_movie(userID,nearest,pivotvalues,neighborCount=5,moviesToRecommend=10):
	"""
	nearest is a num_movies x neighbourCount sized matrix giving closest 5 movies for each movie, using movieData
	arr is the rating for each movie by the user
	movierecs contains "score" of each movie. top moviesToRecommend scores are chosen from this array
	score is calculated by:
	taking each movie rated by the user, and incrementing the scores of each of that movies' neighbours by the rating of the 
	movie given by the user
	watched is list of movies rated by user
	recommend is the set to be returned, with watched movies subtracted
	"""
	movierecs = np.zeros(len(nearest))
	arr = pivotvalues[userID,:]
	watched = []
	for i in range(len(arr)):
	    if arr[i]>0:
	        watched.append(i)
	        for j in range(neighborCount):
	            movierecs[nearest[i][j]] = movierecs[nearest[i][j]] + arr[i]
	recommend = movierecs.argsort()[0:moviesToRecommend]	
	recommend = np.setdiff1d(recommend,watched)
	return recommend[0:moviesToRecommend]
print getRecommendations_movie(userID=187,nearest = nearest,pivotvalues = pivot.values)