class als_wr:
	# Call als = als_wr(path = "path", test_size = 0.2)
	def __init__(self, path = '/', test_size = 0.2):
		import os
		import os.path as op
		import numpy as np
		os.chdir(path)
		outfile_test = 'test_mat.npy'
		outfile_train = 'train_mat.npy'
		reimport = True
		if op.isfile(outfile_test) & op.isfile(outfile_train):
			reimport = False
			print('Importing previously imported data...')
			self.test = np.load(outfile_test)
			self.train = np.load(outfile_train)
		if reimport:
			from utils import import100k
			self.train, self.test = import100k(test_size=test_size)
		print('Done importing.')
		
	'''
	Call als.fit()
	'''
	def fit(self, n_factors = 8, n_iterations = 10, lambda_ = 10):
	    print('Running fit...')
	    import numpy as np
	    import os.path as op
	    m, n = self.train.shape
	    Q = self.train
	    W = Q > 0.5
	    W[W == True] = 1
	    W[W == False] = 0
	    outfile_X = 'X.npy'
	    outfile_Y = 'Y.npy'
	    if op.isfile(outfile_X) & op.isfile(outfile_Y):
	    	print('X and Y initialzied from before.')
	    	X = np.load(outfile_X)
	    	Y = np.load(outfile_Y)
	    else:
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
	        np.save(outfile_X, X)
	        np.save(outfile_Y, Y)
	    self.prediction = np.dot(X,Y)
	    self.X = X
	    self.Y = Y
	    self.check_error()
	    print('Done.')
	'''
	Gets automatically called.
	'''
	def check_error(self):
		import numpy as np
		'''
		W is just a matrix of whether rated or not
		'''
		print('Finding RMSE...')
		W_test = self.test > 0.5
		W_test[W_test == True] = 1
		W_test[W_test == False] = 0
		W_train = self.train > 0.5
		W_train[W_train == True] = 1
		W_train[W_train == False] = 0
		'''
		Finding RMSE
		'''
		self.train_error = np.sqrt(np.sum( np.square(self.train - np.multiply(self.prediction,W_train)) )/np.count_nonzero(self.train))
		self.test_error = np.sqrt(np.sum( np.square(self.test - np.multiply(self.prediction,W_test)) )/np.count_nonzero(self.test))
		print('Done.')
		print('Train error is :{}'.format(self.train_error))
		print('Test error is :{}'.format(self.test_error))
		
if __name__ == '__main__':
	als = als_wr(test_size = 0.2)
	als.fit(n_iterations = 2)