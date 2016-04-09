"""
Created on Sat Apr 2 01:35:15 2016
Using zeros for empty values, when using my created random matrice, Xnew and Ynew are ok, rmse ~3
when using sample matrice provided by ganesh, Xnew is compltely zero
@author: satvikk
"""
import numpy as np
import nimfa as nf
from scipy.sparse import csr_matrix

def rmse(M):
	M=np.power(M,2)
	M=np.sum(M)/(np.shape(M)[0]*np.shape(M)[1])
	M=np.sqrt(M)
	return M
def rmseQ(M,T):
	M=np.power(M,2)
	M=np.sum(M[T])/np.sum(T)
	M=np.sqrt(M)
	return M

X = np.random.rand(10,5)  #randomly generated
Y = np.random.rand(5,20)
X = (X*5).astype(int) +1
Y = (Y*5).astype(int) +1

#X = np.loadtxt("X_gen.txt")  #sample matrice by Ganesh
#Y = np.loadtxt("Y_gen.txt")
#print X 
#print Y
T = np.random.rand(10,20)
T = T>0.9
Q = (np.dot(X,Y)).astype(float)		#matice is 30% sparse
Q[T==False] = 0
#print Q
#print T
#print Q[T]
sparseQ = csr_matrix(Q)

L =  nf.Snmf(sparseQ,seed="random_vcol",rank=5,max_iter=50) 
Lmod = L.factorize()
Xnew = L.basis().todense()
Ynew = L.coef().todense()
Qnew = L.fitted().todense()
Qx = Q - Qnew
Xx = X - Xnew
Yx = Y - Ynew
print X
print Y
print Ynew.astype(int)
print Xnew.astype(int)
print Qx.astype(int)
print rmseQ(Qx,T)
print Qx[T]
print np.sum(Q[T])/np.sum(T)
#print Qnew[T]- Q[T]
#print rmse(Xnew-X)
#print rmse(Ynew-Y)
#print Qnew
#print Q
#print Xnew.todense()
#print Ynew

#print Qx[T]
#print rmse(Qx)