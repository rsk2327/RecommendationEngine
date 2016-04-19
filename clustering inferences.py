import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from scipy.sparse import csr_matrix
import scipy.cluster.vq

os.chdir("/home/satvik/Analytics/Recommender Project")
itemDat = pd.read_table("ml-100k/u.item",sep="|",header=None)
itemDat.columns = ['movieID', 'movie title' , 'release date' , 'video release date' ,
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Childrens' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
             ' Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
D = itemDat.loc[:,'Action':'Western'].values
#D = scipy.cluster.vq.whiten(D)#feature-scale
#%% Plot Cost vs num_clusters
distortion =[]
for num_clusters in range(2,10):
    distortion.append(scipy.cluster.vq.kmeans(D,num_clusters,iter=1000)[1])
    #idx,_=scipy.cluster.vq.vq(D,centroids)#each movie is in a cluster
print distortion
plt.plot(distortion)
plt.title("Cost of Clusters")
plt.xlabel("num_clusters")
plt.ylabel("Distortion")
plt.show()
#%% Plot histograms of cluster densities
num_clusters=5
centroids,distortion=scipy.cluster.vq.kmeans(D,num_clusters,iter=100)
idx,_=scipy.cluster.vq.vq(D,centroids)#each movie is in a cluster
plt.hist(idx,bins=num_clusters)
plt.title("Cluster Densities, for "+str(num_clusters)+" clusters")
plt.xlabel("Cluster Number")
plt.ylabel("No. Movies")
plt.show()
print idx
u = np.zeros((num_clusters,18))
for i in range(num_clusters):
    T = idx==i
    E=D[T,:]
    E = np.sum(E,axis=0)
    u[i,:]=E
u=u.astype(int)
print u

#%% Bar charts of individual clusters to movie features
l=len(itemDat.columns)
for i in range(0,num_clusters):
    plt.figure()
    print len(u[i,:])
    plt.bar(np.arange(18),u[i,:])
    plt.title("CLuster"+str(i))
    plt.xlabel("Movie feature")
    plt.ylabel("No. Movies")
    plt.show()

#%% Generate Covariance matrix of feature vs feature for different clusters 
num_clusters=6
centroids,distortion=scipy.cluster.vq.kmeans(D,num_clusters,iter=100)
u = np.zeros((num_clusters,18))
for i in range(num_clusters):
    T = idx==i
    E=D[T,:]
    E = np.sum(E,axis=0)
    u[i,:]=E
u=u.astype(int)
print u
print np.sum(u,axis=0)
covu = np.cov(u.transpose())
covu = np.triu(covu)
covu=covu.astype(int)
#print covu.astype(int)
print covu
np.savetxt('CovarianceCluster.txt',covu)

