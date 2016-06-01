import os
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix


from sklearn.metrics import r2_score
def ALS_algo(R,W,n_factors=8,lambda_=10,n_iterations=10):
    Q=R
    m, n = Q.shape
    X = 5 * np.random.rand(m, n_factors) 
    Y = 5 * np.random.rand(n_factors, n)

    print "Starting iterations..."
    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                                    np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                    np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
        print('{}th iteration is completed of {}'.format(ii + 1,n_iterations))
        # TEMP=get_error(Q, X, Y, W)
        # print('RMS Error on Training Set after {}th iteration is {}'.format(ii + 1,TEMP))
        # print('Saving current X and Y...')
        # np.savetxt('X.txt', X);
        # np.savetxt('Y.txt', Y);
        # print('Saved.')
    weighted_Q_hat = np.dot(X,Y)
    return weighted_Q_hat,X,Y
