import pandas as pd
import numpy as np
import os

def get_error(Q, X, Y, W):
    TEMP=np.sum((W * (Q - np.dot(X, Y)))**2)
    TEMP=TEMP/np.sum(W)
    TEMP=TEMP**0.5
    return TEMP

def train_test_split(ratings):
    test = np.zeros(ratings.shape)
    train = ratings.copy()
    W=train>0.5
    W[W==True]=1
    W[W==False]=0
    for user in xrange(ratings.shape[0]):
        test_size=round(0.2*np.sum(W[user,:]))
        test_ratings = np.random.choice(ratings[user, :].nonzero()[0],size=test_size,replace=False)
        train[user, test_ratings] = 0.
        test[user, test_ratings] = ratings[user, test_ratings]
        
    # Test and training are truly disjoint
    assert(np.all((train * test) == 0)) 
    return train, test

print ('Running Code...')
print ('Impoting data...')
movie_headers = ['movie_id', 'title', 'genres']
movies = pd.read_table('1m/movies.dat',
                       sep='::', header=None, names=movie_headers)
movie_titles = movies.title.tolist()

rating_headers = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('1m/ratings.dat', sep='::', header=None, names=rating_headers)

df=ratings.join(movies,on = 'movie_id', rsuffix='_r')
del df['movie_id_r']

rp=df.pivot_table(columns='movie_id',index='user_id',values='rating')
rp=rp.fillna(0);
print('Done importing data.')
print('Splitting Data...')
ratings=rp.values

train,test=train_test_split(ratings)

Q=train
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

print "Starting iterations..."
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

W_test=test>0.5
W_test[W_test==True] = 1
W_test[W_test==False] = 0
TEMP=get_error(test, X, Y, W_test)

print('RMS Error on Test Set is {}'.format(TEMP))
'''
def print_recommendations(W=W, Q=Q, Q_hat=weighted_Q_hat, movie_titles=movie_titles,jj=5):
    Q_hat -= np.min(Q_hat)
    Q_hat *= float(5) / np.max(Q_hat)
    movie_ids = np.argmax(Q_hat - 5 * W, axis=1)
    movie_ids = movie_ids[jj]
    print('User {} liked {}'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
    print('User {} did not like {}'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
    print('User {} recommended movie is {} - with predicted rating: {}'.format(jj + 1, movie_titles[movie_ids], Q_hat[jj, movie_ids]))
print_recommendations(jj=13)
'''
