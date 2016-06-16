# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 20:46:23 2016

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
from scipy.sparse import csr_matrix
import pylab

print('Importing data.')
ratingData,movieData,userData = importData()

#%%
#Combining the ratings data with movie and user features

data = pd.merge(ratingData,userData,how="left",on="userID")
data = pd.merge(data,movieData,how="left",on="movieID")

data = cleanData(data)  #removes unneccesary rows and columns
data = dateTime(data)

data.head()

#%%

#To-Do : Need to include ALS variable

categorical_variables = ["gender", "occupation"]
for variable in categorical_variables:
    dummies = pd.get_dummies(data[variable], prefix=variable)
    data = pd.concat([data, dummies], axis=1)
    data.drop([variable], axis=1, inplace=True)


print('Splitting data.')    
train,test = train_test_split(data,test_size=0.2,random_state=1)
#%%
n_u = len(ratingData['userID'].cat.categories)
n_m = len(ratingData['movieID'].cat.categories)
test_col = np.array(test['userID'].values)-1
test_row = np.array(test['movieID'].values)-1
test_dat = np.array(test['rating'].values)
train_col = np.array(train['userID'].values)-1
train_row = np.array(train['movieID'].values)-1
train_dat = np.array(train['rating'].values)
train_mat = (csr_matrix((train_dat, (train_row, train_col)), shape=(n_m, n_u)).toarray()).T
test_mat = (csr_matrix((test_dat, (test_row, test_col)), shape=(n_m, n_u)).toarray()).T
#%%
print('Running ALS.')
prediction,user_features,movie_features = als(train_mat)
testRMSE = RMSE_matrix(prediction, test_mat)
trainRMSE = RMSE_matrix(prediction, train_mat)
print "ALS\nTrain RMSE : %f  Test RMSE : %f"%(trainRMSE, testRMSE)

data['movieID'] = data['movieID'].astype('int')
data['userID'] = data['userID'].astype('int')
features = [ u'timestamp', u'age',
       u'unknown', u'Action',
       u'Adventure', u'Animation', u'Childrens', u'Comedy', u'Crime',
       u'Documentary', u'Drama', u'Fantasy', u' Film-Noir', u'Horror',
       u'Musical', u'Mystery', u'Romance', u'Sci-Fi', u'Thriller', u'War',
       u'Western', u'year', u'month', u'day', u'hour', u'weekday',
       u'releaseYear', u'yearDiff', u'gender_F', u'gender_M',
       u'occupation_administrator', u'occupation_artist', u'occupation_doctor',
       u'occupation_educator', u'occupation_engineer',
       u'occupation_entertainment', u'occupation_executive',
       u'occupation_healthcare', u'occupation_homemaker', u'occupation_lawyer',
       u'occupation_librarian', u'occupation_marketing', u'occupation_none',
       u'occupation_other', u'occupation_programmer', u'occupation_retired',
       u'occupation_salesman', u'occupation_scientist', u'occupation_student',
       u'occupation_technician', u'occupation_writer']
data['ALS'] = prediction[data.loc[:, 'userID']-1, data.loc[:, 'movieID']-1]
features.append('ALS')
for i in range(len(movie_features)):
    data["UserFeature{}".format(i)] = user_features[data.loc[:, 'userID']-1, i]
    features.append("UserFeature{}".format(i))
    data["MovieFeature{}".format(i)] = movie_features[i, data.loc[:, 'movieID']-1]
    features.append("MovieFeature{}".format(i))

train,test = train_test_split(data,test_size=0.2,random_state=1)
ytrain = train.pop('rating')
ytest = test.pop('rating')

########## RandomForest model #############           
print('Training Random Forest.')
model = RandomForestRegressor(100, oob_score=True,random_state=42, n_jobs=-1)
model.fit(train[features],ytrain)

trainRMSE = RMSE(ytrain,model.predict(train[features]))
testRMSE = RMSE(ytest,model.predict(test[features]))
print "Train RMSE : %f  Test RMSE : %f"%(trainRMSE, testRMSE)

#%%
feature_importances = pd.Series(model.feature_importances_,index=train[features].columns)
feature_importances.sort(ascending=False)
few_features = feature_importances[0:12]
few_features.plot(kind="barh",figsize=(7,6))
pylab.show()

#%%##### XGBoost ########################
print('Training XGBoost.')
model = xgb.XGBRegressor(max_depth=7,
                         learning_rate=0.1,
                         n_estimators=100,
                         silent= False,
                         nthread=-1)
model.fit(train[features] ,ytrain)

trainPred,testPred = model.predict(train[features]),model.predict(test[features])
trainRMSE,testRMSE = RMSE(ytrain,trainPred) , RMSE(ytest,testPred)
print "Train RMSE : %f  Test RMSE : %f"%(trainRMSE, testRMSE)
#%%

xgb.plot_importance(model)
