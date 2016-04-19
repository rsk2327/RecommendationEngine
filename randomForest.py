from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import pylab
import os

# X.describe() shows all numerical variables
# X.head shows all numerical and categorical variables

# The Out of Box Score of with and without 
# OOB score = 0.425379456052
# OOB score = 0.427048611386

print ('Importing data...')
os.chdir("C://Users/Ganesh Anand/Desktop/Recommendation Engine")
X = pd.read_csv("combinedData.csv")
X.describe()
#After checking the columns I found that the indexes for ratings were showing up here
X = X.drop(X.columns[0],axis=1)
#Removed release dates as I was unsure how to use it
X = X.drop(['release date','zipcode'],axis=1)
y = X.pop('rating')
print ('Data imported.')

numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
X[numeric_variables].head()
#I have 2 categorical variables left occupation and gender

print('Creating Random Forest and running fit method...')
model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
model.fit(X[numeric_variables],y)
print('Tasks completed.')
print('Initial benchmark done.')
print('OOB score = {}'.format(model.oob_score_))
#Initial model done.

categorical_variables = ["gender", "occupation"]
for variable in categorical_variables:
    #X[variable].fillna("Missing", inplace=True)
    dummies = pd.get_dummies(X[variable], prefix=variable)
    X = pd.concat([X, dummies], axis=1)
    X.drop([variable], axis=1, inplace=True)

print('Done building variables.\nBuilding model...')
model = RandomForestRegressor(100, oob_score=True, random_state=42)
model.fit(X,y)
print('Second model built and benchmark done.')
print('OOB score = {}'.format(model.oob_score_))

feature_importances = pd.Series(model.feature_importances_,index=X.columns)
feature_importances.sort(ascending=False)
few_features = feature_importances[0:12]
few_features.plot(kind="barh",figsize=(7,6))
pylab.show()
