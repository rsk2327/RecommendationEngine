import datetime
import pandas as pd
import numpy as np

def als(train):
	n_factors = 8
	n_iterations = 15
	lambda_ = 10
	m, n = train.shape
	Q = train
	W = Q > 0.5
	W = W.astype(int)
	print('X and Y randomly initialzied.')
	X = 5 * np.random.rand(m, n_factors) 
	Y = 5 * np.random.rand(n_factors, n)
	for ii in range(n_iterations):
		for u, Wu in enumerate(W):
			X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
	                                np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
		for i, Wi in enumerate(W.T):
			Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
	                                np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
		print('{}th iteration is completed of {}'.format(ii + 1,n_iterations))
	prediction = np.dot(X,Y)
	print('Done.')
	return prediction, X, Y

def importData(dataDir=''):
    """
    Imports u.data, u.item and u.user and returns it in DataFrame format
    as ratingData, movieData and userData
    """
    ratingData = pd.read_table(dataDir+"ml-100k/u.data",sep="\t",header=None)
    ratingData.columns=['userID','movieID','rating','timestamp']
    ratingData['userID'] = ratingData['userID'].astype("category")            #converting into categorical variables
    ratingData['movieID'] = ratingData['movieID'].astype('category')

    movieData = pd.read_table(dataDir+"ml-100k/u.item",sep="|",header=None)
    movieData.columns = ['movieID', 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
             ' Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
              
    userData = pd.read_table(dataDir+"ml-100k/u.user",sep="|",header=None)
    userData.columns=['userID','age','gender','occupation','zipcode']
    
    return (ratingData,movieData,userData)

def cleanData(data):
    """
    Pre-process the data. Variable type changes, missing value removal etc.
    """
    data = data.drop(data.index[[2172, 3781, 7245, 12475, 14756, 15292, 49295, 93523, 99723]],axis=0,inplace=False)
    data = data.drop(["video release date", 'IMDb URL'],axis=1)
    
    data.userID = data.userID.astype("category")
    data.movieID = data.movieID.astype("category")
    data.gender = data.gender.astype("category")
    data.occupation = data.occupation.astype("category")
    
	#data = dateTime(data)
    
    return data

def dateTime(data):
    val = data.timestamp.apply(datetime.datetime.fromtimestamp)
    
    data['year'] = val.apply(lambda x : x.year)
    data['month'] = val.apply(lambda x : x.year)
    data['day'] = val.apply(lambda x : x.year)
    data['hour'] = val.apply(lambda x : x.year)
    data['weekday'] = val.apply(datetime.datetime.weekday)
    
    data['release date'] = data['release date'].apply(strpReleaseDate)
    data['releaseYear'] = data['release date'].apply(lambda x : x.year)
    data['yearDiff'] = (data.year-data.releaseYear)
    
    return data
    
def strpReleaseDate(x):
    x = str(x)
    return datetime.datetime.strptime(x,'%d-%b-%Y')

def RMSE(y_pred,y_true):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def RMSE_matrix(prediction, test_mat):
	mask = test_mat>0;	mask[mask==True]=1; mask[mask==False]=0
	prediction = np.multiply(prediction, mask)
	return (np.sum((prediction - test_mat)**2)/np.sum(mask))**0.5
