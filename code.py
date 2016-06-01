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
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pylab
import time
from ganeshALS import ALS_algo
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

def createDataFrame(ratingsDat,itemDat,userDat,predictedR,X,Y): #to create DataFrame for randomForest
    D = pd.merge(ratingsDat,itemDat,on="movieID")
    D = pd.merge(D,userDat,on="userID")
    D = D.drop(D.ix[:,['movie title','video release date','IMDb URL']].head(0).columns,axis=1)
    D['ALS'] = predictedR[D.loc[:,'userID']-1 , D.loc[:,'movieID']-1]

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
    #movie 267 is a bad data and should be removed from the system
    D.drop(D.index[[1711, 4776, 17957, 20011, 21645, 32870, 34295, 42528, 53849]],inplace=True)
    y = lambda x: str(x).split('-')[0]
    D['release day']=D['release date'].apply(y) 
    
    months={'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    y = lambda x: months[str(x).split('-')[1]]
    D['release month']=D['release date'].apply(y)
    y = lambda x: str(x).split('-')[2]
    D['release year']=D['release date'].apply(y)
    D.to_csv('combinedData.csv')        #saving

def timestamp():#adding time of rating, in human redable format
    D = pd.read_csv("combinedData.csv")
    y= lambda x: time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.localtime(x))
    D['time']=D['timestamp'].apply(y)
    print D['time']
    D.to_csv('combinedData.csv')

#%%
def RandomForest():
    print ('Importing data...')
    X = pd.read_csv("combinedData.csv")
    #After checking the columns I found that the indexes for ratings were showing up here
    X = X.drop(X.columns[0],axis=1)
    X = X.drop(['release date','zipcode'],axis=1)
    print ('Data imported.')
    
    categorical_variables = ["gender", "occupation"]
    for variable in categorical_variables:
        #X[variable].fillna("Missing", inplace=True)
        dummies = pd.get_dummies(X[variable], prefix=variable)
        X = pd.concat([X, dummies], axis=1)
        X.drop([variable], axis=1, inplace=True)
    
    train,test = train_test_split(X,test_size=0.2,random_state=1)
    ytrain = train.pop('rating')
    ytest = test.pop('rating')
    print train.columns
    
    print('Done building variables.\nBuilding model...')
    features = ['ALS','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary'
               ,'Drama','Fantasy',' Film-Noir','Horror','Musical','Mystery'
               ,'Romance','Sci-Fi','Thriller','War','Western','cluster'
               ,'age', u'gender_F', u'gender_M',u'gender_F', u'gender_M',
           u'occupation_administrator', u'occupation_artist', u'occupation_doctor',
           u'occupation_educator', u'occupation_engineer',
           u'occupation_entertainment', u'occupation_executive',
           u'occupation_healthcare', u'occupation_homemaker', u'occupation_lawyer',
           u'occupation_librarian', u'occupation_marketing', u'occupation_none',
           u'occupation_other', u'occupation_programmer', u'occupation_retired',
           u'occupation_salesman', u'occupation_scientist', u'occupation_student',
           u'occupation_technician', u'occupation_writer']
           
    model = RandomForestRegressor(100, oob_score=True, random_state=42, n_jobs=-1)
    model.fit(train[features],ytrain)
    print('Model built.\nRunning benchmarks...')
    r2 = r2_score(ytest, model.predict(test[features]))
    rmse = np.sqrt(np.mean((ytest - model.predict(test[features]))**2))
    
    feature_importances = pd.Series(model.feature_importances_,index=train[features].columns)
    feature_importances.sort(ascending=False)
    few_features = feature_importances[0:12]
    few_features.plot(kind="barh",figsize=(7,6))
    pylab.show()
    
    print('R2 score is {}'.format(r2))
    print('RMSE score is {}'.format(rmse)) 
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


print ('Saving combined dataframe for random forest...')
createDataFrame(ratingsDat,itemDat,userDat,predictedR,X,Y)
appendDataFrame()
timestamp()
print ('Data saved.')
