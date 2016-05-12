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
#import nimfa as nf
#%%
def get_train_error(trueRating, predRating, W, rmse=False):            #Computes error for the training dataset
    
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
import time
def createDataFrame(ratingsDat,itemDat,userDat,predictedR,X,Y): #to create DataFrame for randomForest
    D = pd.merge(ratingsDat,itemDat,on="movieID")
    D = pd.merge(D,userDat,on="userID")
    D = D.drop(D.ix[:,['movie title','video release date','IMDb URL']].head(0).columns,axis=1)
    D['ALS'] = predictedR[D.loc[:,'userID']-1 , D.loc[:,'movieID']-1]
    y = lambda x: x.split('-')[0]
    D['release day'] = D['release date'].apply(y)
    y = lambda x: x.split('-')[1]
    D['release month']=D['release date'].apply(y)
    y = lambda x: x.split('-')[2]
    D['release year']=D['release date'].apply(y)
    for i in range(len(np.transpose(X))):
    	colname = "UserFeature{}".format(i)
    	D[colname] = X[D.loc[:,'userID']-1 , i]
    for i in range(len(Y)):
    	colname = "MovieFeature{}".format(i)
    	D[colname] = Y[i , D.loc[:,'movieID']-1]
    D.to_csv('combinedData.csv')        #saving
    #train,test=train_test_split(D,)
    #print D
def appendDataFrame():#creating columns for release dates
    D = pd.read_csv("combinedData.csv")
    y = lambda x: len(str(x).split('-'))
    D['g'] = D['release date'].apply(y) #found rows with invalid release date and dropped them
    E =  D[D['g']==1].index#movie 267 is a bad data and should be removed from the system
    D.drop(D.index[[1711, 4776, 17957, 20011, 21645, 32870, 34295, 42528, 53849]],inplace=True)
    y = lambda x: str(x).split('-')[0]
    D['release day']=D['release date'].apply(y) 
    
    months={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    y = lambda x: months[str(x).split('-')[1]]
    D['release month']=D['release date'].apply(y)
    y = lambda x: str(x).split('-')[2]
    D['release year']=D['release date'].apply(y)
    D.to_csv('combinedData.csv')        #saving
os.chdir("/home/satvik/Analytics/Recommender Project/RecommendationEngine")
def timestamp():
    D = pd.read_csv("combinedData.csv")
    y= lambda x: time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(x))
    D['time']=D['timestamp'].apply(y)
    print D['time']
    D.to_csv('combinedData.csv')
timestamp()
#%%
def movieKMeans(itemDat):#to reduce the total number of movie features
    #np.set_printoptions(threshold=10)
    #os.chdir("/home/satvik/Analytics/Recommender Project")
    #itemDat = pd.read_table("ml-100k/u.item",sep="|",header=None)
    #itemDat.columns = ['movieID', 'movie title' , 'release date' , 'video release date' ,
    #          'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
    #          'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
    #         ' Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
    #          'Thriller' , 'War' , 'Western']
    #D = scipy.cluster.vq.whiten(D)#feature-scale
    num_clusters =3
    D = itemDat.loc[:,'Action':'Western'].values
    centroids,distortion = scipy.cluster.vq.kmeans(D,num_clusters)#
    idx,_=scipy.cluster.vq.vq(D,centroids)#each movie is in a cluster
    #print idx
    return idx
#%%
def ALS_algo(R,W,n_factors=8,lambda_=10,n_iterations=10):
    Q=R
    m, n = Q.shape
    X = 5 * np.random.rand(m, n_factors) 
    Y = 5 * np.random.rand(n_factors, n)

    print "Starting iterations..."
    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                                    np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                    np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
        print('{}th iteration is completed of {}'.format(ii + 1,n_iterations))
        # TEMP=get_error(Q, X, Y, W)
        # print('RMS Error on Training Set after {}th iteration is {}'.format(ii + 1,TEMP))
        # print('Saving current X and Y...')
        # np.savetxt('X.txt', X);
        # np.savetxt('Y.txt', Y);
        # print('Saved.')
    weighted_Q_hat = np.dot(X,Y)
    return weighted_Q_hat,X,Y



#%%############# IMPORTING THE DATASETS ###################
print ('Importing data...')
os.chdir("/home/satvik/Analytics/Recommender Project")
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
print ('Data Imported.')

print ('Splitting data...')
train,test = train_test_split(ratingsDat,test_size=0.2,random_state=1)
print ('Split done.')
# print ratingsDat

print ('Clustering...')
itemDat['cluster'] = movieKMeans(itemDat)
itemDat['cluster'] = itemDat['cluster'].astype('category')
print ('Done clustering.')

num_users = len(ratingsDat['userID'].cat.categories)
num_movies = len(ratingsDat['movieID'].cat.categories)
###########  CREATING RATINGS MATRIX ###########################
#Creating ratings matrix R and weight matrix W. 
# R : Ratings matrix (only has ratings from the train dataset)
# W : Keep check of which cells have ratings
#%%
print ('Making Ratings and Weight matrix...')
R = np.zeros((num_users,num_movies))
W = np.zeros((num_users,num_movies))  

for i in range(len(train)):
    R[ train.iloc[i]['userID']-1 , train.iloc[i]['movieID']-1 ] = train.iloc[i]['rating']    #userID is 1-based while R matrix is 0-based
    W[ train.iloc[i]['userID']-1 , train.iloc[i]['movieID']-1 ] = 1

print ('Matrices created.')

print ('Running ALS...')
predictedR,X,Y=ALS_algo(R,W,n_factors=8,lambda_=10,n_iterations=10)
# increase the n_iterations and n_factors?
# sparseR = csr_matrix(R)    
# #%%################################################################
# ## ALS implementation using NIMFA package
# als =  nf.Lsnmf(sparseR,seed="random_vcol",rank=100,max_iter=15,beta=0.1)            #Try using different rank (#of features) and see the reduction in training error
# als_fit= als.factorize()
# #%%
# user_features = als_fit.basis()
# movie_features = als_fit.coef()
# print user_features.todense()[1:10][1:10]
# predictedR = np.dot( user_features.todense() , movie_features.todense() )
print ('ALS done.')
#%%
train_err = get_train_error(R,predictedR,W,rmse=True)
#
test_err = get_test_error(test,predictedR,rmse=True)
#%%

print ('Saving combined dataframe for random forest...')
createDataFrame(ratingsDat,itemDat,userDat,predictedR,X,Y)
print ('Data saved.')
