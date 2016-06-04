import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cross_validation import train_test_split

def import100k(path = '/', test_size = 0.2, random_state = 1):
	# Function that imports MovieLens 100k data from given path 
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
