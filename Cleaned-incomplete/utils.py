from scipy.sparse import csr_matrix
import datetime
import pandas as pd
import numpy as np

def alspostprocess(data, prediction, user_features, movie_features, n_features):
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
    categorical_variables = ["gender", "occupation"]
    for variable in categorical_variables:
        dummies = pd.get_dummies(data[variable], prefix=variable)
        data = pd.concat([data, dummies], axis=1)
        data.drop([variable], axis=1, inplace=True)
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
