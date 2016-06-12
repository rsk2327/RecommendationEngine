import datetime
import pandas as pd
import numpy as np

def import100k(path = '/', test_size = 0.2, random_state = 1):
	# Function that imports MovieLens 100k data from given path 
	import os
	import pandas as pd
	import numpy as np
	from scipy.sparse import csr_matrix
	from sklearn.cross_validation import train_test_split
	outfile_test = 'test_mat.npy'
	outfile_train = 'train_mat.npy'
	print('Reimporting data...')
	os.chdir(path)
	ratingsDat = pd.read_table("ml-100k/u.data",sep="\t",header=None)
	ratingsDat.columns = ['userID','movieID','rating','timestamp']
	ratingsDat['userID'] = ratingsDat['userID'].astype("category")
	ratingsDat['movieID'] = ratingsDat['movieID'].astype('category')
	train,test = train_test_split(ratingsDat,test_size=test_size,random_state=random_state)
	n_u = len(ratingsDat['userID'].cat.categories)
	n_m = len(ratingsDat['movieID'].cat.categories)
	test_col = np.array(test['userID'].values)-1
	test_row = np.array(test['movieID'].values)-1
	test_dat = np.array(test['rating'].values)
	train_col = np.array(train['userID'].values)-1
	train_row = np.array(train['movieID'].values)-1
	train_dat = np.array(train['rating'].values)
	train_mat = (csr_matrix((train_dat, (train_row, train_col)), shape=(n_m, n_u)).toarray()).T
	test_mat = (csr_matrix((test_dat, (test_row, test_col)), shape=(n_m, n_u)).toarray()).T
	np.save(outfile_test, test_mat)
	np.save(outfile_train, train_mat)
	return train_mat, test_mat

def importData(dataDir):
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
    data = data.drop("video release date",axis=1)
    
    data.userID = data.userID.astype("category")
    data.movieID = data.movieID.astype("category")
    data.gender = data.gender.astype("category")
    data.occupation = data.occupation.astype("category")
    
    
#    data = dateTime(data)
    
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

def prepareForRF(path = '/', prediction=False, X=False, Y=False):
	outfile_combined = 'combinedData.csv'
	remport = True
	if op.isfile(outfile_combined):
		reimport = False
		data = pd.read_csv(outfile_combined)
	if remport:	
		ratingsDat = pd.read_table("ml-100k/u.data",sep="\t",header=None)
		ratingsDat.columns=['userID','movieID','rating','timestamp']
		ratingsDat['userID'] = ratingsDat['userID'].astype("category")
		ratingsDat['movieID'] = ratingsDat['movieID'].astype('category')

		itemDat = pd.read_table("ml-100k/u.item",sep="|",header=None)
		itemDat.columns = ['movieID', 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
             ' Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']              
		userDat = pd.read_table("ml-100k/u.user",sep="|",header=None)
		userDat.columns = ['userID','age','gender','occupation','zipcode']
		data = pd.merge(ratingsDat, itemDat, on='movieID')
		data = pd.merge(data, userDat, on='userID')
		data = data.drop(data.ix[:,['movie title','video release date','IMDb URL']].head(0).columns,axis=1)
		if not prediction and not X and not Y:
			data['ALS'] = prediction[data.loc[:, 'userID']-1, data.loc[:, 'movieID']-1]
			for i in range(len(Y)):
				data["UserFeature{}".format(i)] = X[data.loc[:, 'userID']-1, i]
				data["MovieFeature{}".format(i)] = Y[i, data.loc[:, 'movieID']-1]
		data.to_csv(outfile_combined)
	return data
    #%%
