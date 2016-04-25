from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import pylab
import os
from sklearn.cross_validation import train_test_split

# X.describe() shows all numerical variables
# X.head shows all numerical and categorical variables
# R2 score is 0.434810008711
# RMSE score is 0.848644536894

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
"""features = ['ALS','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary'
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
"""
features = ['ALS','userID']
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
