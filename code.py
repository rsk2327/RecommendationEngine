# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 00:39:10 2016

@author: rsk
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
import scipy.cluster.vq
import nimfa as nf

def get_train_error( trueRating, predRating, W, rmse=False ):            #Computes error for the training dataset
    
    ss = np.sum( np.square(trueRating - np.multiply(predRating,W)) )
    if rmse==True:
        return np.sqrt(ss/np.sum(W))
    else:
        return ss

def get_test_error( testData, predRating, rmse=False ):
    
    ss=[]
    
    for i in range(len(testData)):
        error = testData.iloc[i]['rating'] - predRating[ testData.iloc[i]['userID']-1 , testData.iloc[i]['movieID']-1 ]
        ss.append(error**2)
    ss = np.sum(ss)
    
    if rmse==True:
        return np.sqrt( ss/len(testData) )
    else:
        return ss
#%%
def createDataFrame(ratingsDat,itemDat,userDat):#,R): #to create DataFrame for randomForest
    D = pd.merge(ratingsDat,itemDat,on="movieID")
    D = pd.merge(D,userDat,on="userID")
    D['ALS']=0
    D = D.drop(D.ix[:,['movie title','video release date','IMDb URL']].head(0).columns,axis=1)
    #D['ALS'] = R[D.loc[:,'userID']-1 , D.loc[:,'movieID']-1]
    D.to_csv('DataFrame.csv')        #saving
    #print D
#%%
def movieKMeans(itemDat):#to reduce the total number of movie features
    num_clusters =7
    D = itemDat.loc[:,'Action':].values
    D = scipy.cluster.vq.whiten(D)#feature-scale
    centroids,distortion = scipy.cluster.vq.kmeans(D,num_clusters)#
    idx,_=scipy.cluster.vq.vq(D,centroids)#each movie is in a cluster
    print idx
############# IMPORTING THE DATASETS ###################

os.chdir("/home/rsk/Documents/RecommenderProject/")
ratingsDat = pd.read_table("ml-100k/u.data",sep="\t",header=None)
ratingsDat.columns=['userID','movieID','rating','timestamp']
ratingsDat['userID'] = ratingsDat['userID'].astype("category")            #converting into categorical variables
ratingsDat['movieID'] = ratingsDat['movieID'].astype('category')

itemDat = pd.read_table("ml-100k/u.item",sep="|",header=None)
itemDat.columns = ['movieID', 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
             ' Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
              
# TO-DO : Convert the genres into categorical variables   
              
userDat = pd.read_table("ml-100k/u.user",sep="|",header=None)
userDat.columns = ['userID','age','gender','occupation','zipcode']

train,test = train_test_split(ratingsDat,test_size=0.2,random_state=1)
#print ratingsDat

#movieKMeans(itemDat)
#%%##########  CREATING RATINGS MATRIX ###########################


num_users = len(ratingsDat['userID'].cat.categories)
num_movies = len(ratingsDat['movieID'].cat.categories)

#Creating ratings matrix R and weight matrix W. 
# R : Ratings matrix (only has ratings from the train dataset)
# W : Keep check of which cells have ratings


R = np.zeros((num_users,num_movies))
W = np.zeros((num_users,num_movies))  

for i in range(len(train)):
    R[ train.iloc[i]['userID']-1 , train.iloc[i]['movieID']-1 ] = train.iloc[i]['rating']    #userID is 1-based while R matrix is 0-based
    W[ train.iloc[i]['userID']-1 , train.iloc[i]['movieID']-1 ] = 1
    
sparseR = csr_matrix(R)    

#%%################################################################

## ALS implementation using NIMFA package
als =  nf.Lsnmf(sparseR,seed="random_vcol",rank=100,max_iter=15,beta=0.1)            #Try using different rank (#of features) and see the reduction in training error
als_fit= als.factorize()


#%%
user_features = als_fit.basis()
movie_features = als_fit.coef()
print user_features.todense()[1:10][1:10]
predictedR = np.dot( user_features.todense() , movie_features.todense() )

get_train_error(R,predictedR,W,rmse=True)
#%%
get_test_error( test,predictedR,rmse=True )
#%%


