import numpy as np

def get_error(Q, X, Y, W):
    TEMP=np.sum((W * (Q - np.dot(X, Y)))**2)
    TEMP=TEMP/np.sum(W)
    TEMP=TEMP**0.5
    return TEMP
#change high value to get more size.
X_gen=np.random.randint(1,high=500,size=(10,5))
Y_gen=np.random.randint(1,high=500,size=(5,20))
'''
X_gen=np.loadtxt('X_gen.txt')
Y_gen=np.loadtxt('Y_gen.txt')
'''
Q=np.dot(X_gen,Y_gen)
W=Q>0.5
W[W==True] = 1
W[W==False] = 0

lambda_ = 10
n_factors = 5
m, n = Q.shape
n_iterations = 15

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

Q_algo=np.dot(X,Y)

print('-'*100)
print(X)
print(X_gen)
print('-'*100)
print(Y)
print(Y_gen)

np.savetxt('X_gen.txt',X_gen)
np.savetxt('Y_gen.txt',Y_gen)


