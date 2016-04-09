#created by Satvik on 26/3/16
#I have used the lsnmf (lease square non negative matrix factorizations method in the nimfa library, it uses ALS
#while a model is created, i am not able to see the output of the model(saved in newM). I am still working on it
#I used rowmeans to fill empty values. not sure what is happening when all row members are nan
#http://nimfa.biolab.si/index.html
#https://github.com/marinkaz/nimfa

import numpy as np
from nimfa import lsnmf
import nimfa
warnings.filterwarnings('ignore')

file = open('./ml-1m/ratings.dat','r')
M = [np.nan]*(6040*3952)
M = np.reshape(M,[6040,3952])
for line in file:
	line = map(int,line.strip().split("::"))
	M[line[0]-1][line[1]-1]=float(line[2])
file.close()
rowMean = np.nanmean(M,1)
colMean = np.nanmean(M,0)
inds=np.where(np.isnan(M))
M[inds] = np.take(rowMean,inds[0])
print "completed step1"
L = lsnmf.Lsnmf(M, max_iter=10, rank=100)
Lmod = L.factorize()
#Lmod = L()
print L.summary()
W = L.basis()
H = L.coef()
newM = L.fitted()
#print newM[1][1193]
#print newM[3297][2622]
#print newM[23][42]

np.savetxt("H.txt", H, newline=" ")
np.savetxt("W.txt", W, newline=" ")
#np.savetxt("NewM.txt", newM, newline=" ")
#print newM[1][1]
#print W
#print H
#print W*H
#print M[1:10][1:10]