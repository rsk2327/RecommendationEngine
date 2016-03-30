import pandas as pd
import numpy as np
import scipy.,arse as ,
import sklearn as sk
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt

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
V_train, V_temp = train_test_split(V, train_size=0.60, random_state=42)
V_test,V_validate=train_test_split(V_temp,test_size=0.5,random_state=42)

W = V_train>0.5
W[W == True] = 1
W[W == False] = 0
# To be consistent with our Q matrix
W = W.astype(np.float64, copy=False)
Q=V_train
lambda_ = 0.1
n_factors = 100
m, n = Q.shape
n_iterations = 20
X = 5 * np.random.rand(m, n_factors) 
Y = 5 * np.random.rand(n_factors, n)
def get_error(Q, X, Y, W):
    return np.sum((W * (Q - np.dot(X, Y)))**2)
weighted_errors = []
for ii in range(n_iterations):
    for u, Wu in enumerate(W):
        X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                               np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
    for i, Wi in enumerate(W.T):
        Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                 np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
    weighted_errors.append(get_error(Q, X, Y, W))
    print('{}th iteration is completed'.format(ii))
weighted_Q_hat = np.dot(X,Y)
#print('Error of rated movies: {}'.format(get_error(Q, X, Y, W)))
def print_recommendations(W=W, Q=Q, Q_hat=Q_hat, movie_titles=movie_titles):
    #Q_hat -= np.min(Q_hat)
    #Q_hat[Q_hat < 1] *= 5
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(5) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
    for jj, movie_id in zip(range(m), movie_ids):
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, movie_titles[movie_id], Q_hat[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')
#print_recommendations()
print_recommendations(Q_hat=weighted_Q_hat)

