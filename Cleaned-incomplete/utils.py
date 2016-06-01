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

def prepareForRF(path = '/', test_size = 0.2):