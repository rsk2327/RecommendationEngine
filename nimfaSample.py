"""
Created on Sat Apr 2 01:35:15 2016

@author: satvikk
While doing the task on creating sample 10x5 and 5x20, I found that using nimfa lsnmf yeilds matrice full of nans,
X and Y are partially nan, Q is completely nan
"""
import numpy as np
import nimfa as nf
from scipy.sparse import csr_matrix

def rmse(Qx):
	Qx = Q-Qnew
	Qx=np.power(Qx,2)
	nnan = ~np.isnan(Qx)
	Qx=np.sum(Qx[nnan])/np.sum(nnan)
	Qx=np.sqrt(Qx)
	return Qx

X = np.random.rand(10,5)
Y = np.random.rand(5,20)
X = (X*5).astype(int) +1
Y = (Y*5).astype(int) +1
#print X 
#print Y
T = np.random.rand(10,20)
T = T>0.3
Q = (np.dot(X,Y)).astype(float)
Q[T==False] = np.nan
#print Q
sparseQ = csr_matrix(Q)

L =  nf.Lsnmf(sparseQ,seed="random_vcol",rank=5,max_iter=10,beta=0.1) 
Lmod = L.factorize()
Xnew = L.basis()
Ynew = L.coef()
Qnew = L.fitted().todense()
#print rmse(Xnew-X)
#print rmse(Ynew-Y)
#print Qnew
#print Q
#print rmse(Q-Qnew)
print Xnew.todense()
#print Ynew