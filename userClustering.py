# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 20:00:34 2016

@author: rsk
"""

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
#%%

dataset = pd.DataFrame(user_features)

from sklearn.cluster import KMeans
num_clusters=2
clust = KMeans(n_clusters=num_clusters,max_iter=300,n_init=200,n_jobs=-1)
clust = clust.fit_predict(dataset)
#%%
dataset["cluster"] = (clust+1.0).astype("int")
dataset["ID"] = list(range(1,944))
dataset["gender"] = userData["gender"]
dataset["occupation"] = userData["occupation"]


#PLOTTING

from ggplot import *
x= np.array(dataset[0])
y= np.array(dataset[1])


p = ggplot(dataset, aes(x=0,y=1,z=2,color="cluster")) +geom_point() +ggtitle("User clusters")
print p

#%%


from sklearn.neighbors import NearestNeighbors
neighborModel = NearestNeighbors()
nearestNeighbors = neighborModel.fit(X=user_features)
nearest= nearestNeighbors.kneighbors(user_features,return_distance=False)


