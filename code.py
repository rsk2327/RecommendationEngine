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
import nimfa as nf

def get_error( trueRating, predRating, W ):            #Computes error for the training dataset
    return np.sum( np.square(trueRating - np.multiply(predRating,W)) )


#%%############## IMPORTING THE DATASETS ###################

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
als =  nf.Lsnmf(sparseR,seed="random_vcol",rank=30,max_iter=8,beta=0.1)            #Try using different rank (#of features) and see the reduction in training error
als_fit= als.factorize()


#%%
user_features = als_fit.basis()
movie_features = als_fit.coef()
predictedR = np.dot( user_features.todense() , movie_features.todense() )

get_error(R,predictedR,W)

