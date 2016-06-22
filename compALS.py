# -*- coding: utf-8 -*-
"""
Created on Wed Jun 22 10:13:15 2016

@author: rsk
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import nimfa as nf


mat=np.array([[2.0,3.0,0.0,0.0,5.0],
              [1.0,0.0,0.0,4.0,2.0],
                [4.0,2.0,0.0,4.0,1.0],
                [1.0,0.0,2.0,0.0,4.0],
                [4.0,5.0,3.0,0.0,0.0]])
                
#%%

m,n = mat.shape
W = (mat>0.5).astype("int")

sparseR = csr_matrix(mat)

#%%
als = nf.Lsnmf(sparseR,seed="random_vcol",rank=4,max_iter=30,beta=0.5)
als_fit = als.factorize()

u = als_fit.basis()
m = als_fit.coef()



predictedR = np.array(np.dot(u.todense(), m.todense()))

for i in range(predictedR.shape[0]):
    for j in range(predictedR.shape[1]):
        predictedR[i][j] = round(predictedR[i][j],2)
        
#%%
RMSE = RMSE_matrix(mat,predictedR)
#%%

def alsPred(matrix=None, sparseness = 0.3, method="lsnmf",n_features=5,n_iterations=30, low=1, high=5):
    
    #generate a random matrix if matrix is not specified
    if isinstance(matrix,tuple) and len(matrix)==2:
        mat = np.random.randint(low=low,high=(high+1),size=matrix)
        mask = np.random.choice([0.0,1.0],size = mat.size,p = [ sparseness , (1.0 - sparseness)]).reshape(mat.shape)
        mat = mat*mask
        print mat
    
    
    if method=="lsnmf":
        
        mask = (mat>0.5).astype("int")
        sparseMat = csr_matrix(mat)
        
        als  = nf.Lsnmf(sparseMat,seed="random_vcol",rank=n_features,max_iter = n_iterations, beta = 0.5)
        als_fit = als.factorize()
        
        U = np.array(als_fit.basis().todense())
        V = np.array(als_fit.coef().todense())
        predictedMat = np.round(np.array(np.dot(U, V)),2)
        
        RMSE = RMSE_matrix(mat,predictedMat)
        return mat,predictedMat,np.round(U,2),np.round(V,2),RMSE
        
    elif method=="wr":
        mask = (mat>0.5).astype("int")        
        output = als(mat,n_factors = n_features,n_iterations = n_iterations)
        RMSE = RMSE_matrix(mat,output[0])
        return mat, output[0],output[1],output[2],RMSE
        
    return 0
        
        
        