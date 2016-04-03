import pandas as pd
import numpy as np
import scipy.sparse as sp
import sklearn as sk
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from numpy import mean,cov,double,cumsum,dot,linalg,array,rank
from pylab import plot,subplot,axis,stem,show,figure
import warnings
warnings.filterwarnings('ignore')

def princomp(A):
 """ performs principal components analysis 
     (PCA) on the n-by-p data matrix A
     Rows of A correspond to observations, columns to variables. 

 Returns :  
  coeff :
    is a p-by-p matrix, each column containing coefficients 
    for one principal component.
  score : 
    the principal component scores; that is, the representation 
    of A in the principal component space. Rows of SCORE 
    correspond to observations, columns to components.
  latent : 
    a vector containing the eigenvalues 
    of the covariance matrix of A.
 """
 # computing eigenvalues and eigenvectors of covariance matrix
 M = (A-mean(A.T,axis=1)).T # subtract the mean (along columns)
 [latent,coeff] = linalg.eig(cov(M)) # attention:not always sorted
 score = dot(coeff.T,M)
 M=dot(coeff.T,score) # projection of the data in the new space
 return coeff,score,M

data_file=pd.read_table('ratings.dat',sep='::',header=None)
users = np.unique(data_file[0])
movies = np.unique(data_file[1])
 
number_of_rows = len(users) #6040
number_of_columns = len(movies) #3706

movie_indices, user_indices = {}, {}
 
for i in range(len(movies)):
    movie_indices[movies[i]] = i
    
for i in range(len(users)):
    user_indices[users[i]] = i
#scipy sparse matrix to store the 1M matrix
V = np.zeros((number_of_rows, number_of_columns))
theta= np.zeros((4832, 2))

#adds data into the sparse matrix
for line in data_file.values:
    u, i , r , gona = map(int,line)
    V[user_indices[u], movie_indices[i]] = r
#V=np.matrix.transpose(V)
#rowMean = V.sum(1) / (V != np.nan).sum(1)
#print (rowMean.shape)
print (V.shape)
V_train, V_test = train_test_split(V, test_size=0.20, random_state=42)
V_train=np.matrix.transpose(V_train) #No. of movies * no. of users
print (V_train.shape)
for i in range(1,2):
	#for i in range(len(V)):
        #V[:,i]-=rowMean
		#V_test,V_validate=train_test_split(V_temp,test_size=0.5,random_state=42)
		results,coeff,M=princomp(V_train.T)
		where_are_NaNs = np.isnan(V_train)
		V_train[where_are_NaNs] = M[where_are_NaNs]
print (V_train)
x=coeff[:,:2]
print(V_train.shape)
print(x.shape)
inn=np.linalg.pinv(x);
print(inn.shape)
print(where_are_NaNs.shape)
print(M.shape)

for i in range(1,4832):
	Y=V_train[:,i];
	theta[i,:]=np.dot(inn,Y);
x=np.matrix.transpose(x); 
Ypred=np.dot(theta,x);
Ypred=np.matrix.transpose(Ypred);
print(V_train.shape)
print(Ypred.shape)
TEMP=np.sum((V_train - Ypred)**2)
TEMP=TEMP/(number_of_columns*number_of_rows)
TEMP=TEMP**0.5
print (np.abs(TEMP))
    





