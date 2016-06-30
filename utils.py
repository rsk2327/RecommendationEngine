from scipy.sparse import csr_matrix
import datetime
import pandas as pd
import numpy as np
import os.path as op

class collab_filter(object):
    def __init__(self):
        pass

    def fit(self, train, neighbours=-1):
        if neighbours==-1:
            neighbours = len(train)-1
        self.train = train
        print('\nRunning collaborative filtering...')
        if op.isfile('Distance_Matrix.npy'):
            self.euclidean = np.load('Distance_Matrix.npy')
        else:
            euclidean = distance.cdist(train, train, 'euclidean')
            euclidean = 1.0/euclidean
            euclidean[np.isinf(euclidean)] = 0
            self.euclidean = euclidean
            np.save('Distance_Matrix.npy', euclidean)
        print('Filling out other ratings...')
        prediction = np.zeros(train.shape)
        for i in range(len(self.euclidean)):
            if i%200==0:
                print('{}% of ratings matrix populated.'.format(100*i/len(self.euclidean)))
            mask = np.ones((len(train),), dtype=bool)
            sim = np.argsort(-self.euclidean[i])
            cut_off = self.euclidean[i, sim[neighbours]]
            mask[i] = False
            weights = self.euclidean[i][mask]
            weights[weights<cut_off]=0
            other_ratings = train[mask]
            weights = repmat(weights, len(train.T), 1).T
            weights = np.multiply(weights, train[mask] > 0)
            user_ratings = np.sum(np.multiply(weights, other_ratings), axis=0)/np.sum(weights, axis=0)
            prediction[i] = user_ratings
        prediction[np.isnan(prediction)] = np.mean(train)
        self.prediction = prediction

    def predict(self):
        return self.prediction

def als_recommend(user_row, Y):
	""" Returns a sorted list of recommendations of movies and the predicted row of ratings """
	Wu = (user_row > 0).astype(int)
	lambda_ = 10
	n_factors = len(Y)
	Xu = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
	                                np.dot(Y, np.dot(np.diag(Wu), user_row.T))).T
	predicted_row = np.dot(Xu, Y)
	recommendations = np.argsort(-np.multiply(predicted_row, (1 - Wu)))
	# Print the names?
	return predicted_row, recommendations

def alspostprocess(data, prediction, features, user_features, movie_features, n_features=10):
    """
    Adds ALS values obtained from ALS decomposition of ratings matrix as
    feature to dataFrame.
    Also adds movie and user features obtained ALS decomposition to 
    corresponding samples

    INPUT     
    data : DataFrame to which ALS, user_features and movie_features 
           are to be added.
           
    prediction : Numpy array containing ALS prediction
    
    user_features : ALS user features
    
    movie_features : ALS movie features
    
    n_features : Number of user_features and movie_features to be added to the DataFrame
    
    OUTPUT
    data : DataFrame with ALS, user_features and movie_features added
    
    features : List of feature names to be used for further modelling
    """
    

    data['ALS'] = prediction[data.loc[:, 'userID']-1, data.loc[:, 'movieID']-1]
    features.append('ALS')
    
    total_features = len(movie_features)
    if n_features>total_features:
        n_features = total_features
        
    for i in range(n_features):
        data["UserFeature{}".format(i)] = user_features[data.loc[:, 'userID']-1, i]
        features.append("UserFeature{}".format(i))
        data["MovieFeature{}".format(i)] = movie_features[i, data.loc[:, 'movieID']-1]
        features.append("MovieFeature{}".format(i))
    return data, features

def alspreprocess(ratingData, test, train):
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
    return test_mat, train_mat

def als(matrix, n_factors=8,n_iterations=15, lambda_=10):
	"""
     Carries out ALS decomposition of a given matrix and returns
     the predicted matrix and the decomposed matrices X and Y
     
     INPUT
     matrix : Numpy matrix to be decomposed
     n_factors : Number of features for the decomposed matrices
     lambda : Regularization factor
     
     OUTPUT
     prediction : ALS predicted matrix i.e product of X.T and Y
     X,Y : Decomposed matrices computed by ALS
 
     """
	m, n = matrix.shape
	Q = matrix
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

def importData(dataDir='',dataset= "ml-100k"):
    """
    Imports u.data, u.item and u.user and returns it in DataFrame format
    as ratingData, movieData and userData
    """
    ratingData = pd.read_table(dataDir+dataset+"/u.data",sep="\t",header=None)
    ratingData.columns=['userID','movieID','rating','timestamp']
    ratingData['userID'] = ratingData['userID'].astype("category")            #converting into categorical variables
    ratingData['movieID'] = ratingData['movieID'].astype('category')

    movieData = pd.read_table(dataDir+dataset+"/u.item",sep="|",header=None)
    movieData.columns = ['movieID', 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
             ' Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
              
    userData = pd.read_table(dataDir+dataset+"/u.user",sep="|",header=None)
    userData.columns=['userID','age','gender','occupation','zipcode']
    
    return (ratingData,movieData,userData)

def cleanData(data):
    """
    Pre-process the data. Variable type changes, missing value removal etc.
    """
    
    indexes = data.index[pd.isnull(data["release date"])]
    # data = data.drop(data.index[[2172, 3781, 7245, 12475, 14756, 15292, 49295, 93523, 99723]],axis=0,inplace=False)
    data = data.drop(indexes,axis=0,inplace=False)
    data = data.drop(["video release date", 'IMDb URL'],axis=1)
    
    
    data.gender = data.gender.astype("category")
    data.occupation = data.occupation.astype("category")
    
    categorical_variables = ["gender", "occupation"]
    for variable in categorical_variables:
        dummies = pd.get_dummies(data[variable], prefix=variable)
        data = pd.concat([data, dummies], axis=1)
        data.drop([variable], axis=1, inplace=True)
    data['movieID'] = data['movieID'].astype('int')
    data['userID'] = data['userID'].astype('int')
    
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
