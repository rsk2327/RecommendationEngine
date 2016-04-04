import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

def get_error(Q, X, Y, W):
    TEMP=np.sum((W * (Q - np.dot(X, Y)))**2)
    TEMP=TEMP/np.sum(W)
    TEMP=TEMP**0.5
    return TEMP


data_file=pd.read_table('ratings.dat',sep='::',header=None)
users = np.unique(data_file[0])
movies = np.unique(data_file[1])
 
number_of_rows = len(users)
number_of_columns = len(movies)

movie_indices, user_indices = {}, {}
 
for i in range(len(movies)):
    movie_indices[movies[i]] = i
    
for i in range(len(users)):
    user_indices[users[i]] = i
#scipy sparse matrix to store the 1M matrix
V = np.zeros((number_of_rows, number_of_columns))

#adds data into the sparse matrix
for line in data_file.values:
    u, i , r , gona = map(int,line)
    V[user_indices[u], movie_indices[i]] = r
V=np.matrix.transpose(V)
rowMean = V.sum(1) / (V != np.nan).sum(1)
print (rowMean.shape)
print (V.shape)
for i in range(len(V)):
        V[:,i]-=rowMean
V_train, V_test = train_test_split(V, test_size=0.20, random_state=42)


Q=V_train
W=Q>0.5
W[W==True] = 1
W[W==False] = 0
print('Done splitting.')
lambda_ = 10
n_factors = 8
m, n = Q.shape
n_iterations = 10

X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)

print ("Starting iterations...")
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    print('{}th iteration is completed of {}'.format(ii + 1,n_iterations))
    TEMP=get_error(Q, X, Y, W)
    print('RMS Error on Training Set after {}th iteration is {}'.format(ii + 1,TEMP))
    print('Saving current X and Y...')
    np.savetxt('X.txt', X);
    np.savetxt('Y.txt', Y);
    print('Saved.')
weighted_Q_hat = np.dot(X,Y)

W_test=V_test>0.5
W_test[W_test==True] = 1
W_test[W_test==False] = 0
TEMP=get_error(V_test, X, Y, W_test)

print('RMS Error on Test Set is {}'.format(TEMP))