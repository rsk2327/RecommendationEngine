from utils import *
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

ratingData,movieData,userData = importData("/home/satvik/Analytics/Recommender Project/")

movieData = movieData.drop(["video release date", 'IMDb URL','movieID','movie title','release date','unknown'],axis=1)
#%%
num_clusters=4
clust = KMeans(n_clusters=num_clusters,max_iter=300,n_init=200,n_jobs=-1)
clust = clust.fit_predict(movieData)

#%%
movieData['cluster']=clust
#%%
M=np.zeros([num_clusters,19])
for i in range(num_clusters):
    M[i,:] = np.array(movieData[movieData.cluster==i].sum())
M = M[:,0:18]
#%%
colnames = movieData.columns.values[0:18]
#%%
N = num_clusters
menMeans = M[:,0]
womenMeans = M[:,1]

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence
colors = ['b','g','r','c','m','y','k','paleturquoise','purple','salmon','seashell','tan'
            'navy','palegreen','orange','lime','olive','orangered']
p=[]

for i in range(18):
    if i==0:
        p.append(plt.bar(ind, M[:,i], width,color=colors[i]))
    else:
        p.append(plt.bar(ind, M[:,i], width,bottom= M[:,i-1],color=colors[i]))

plt.ylabel('movies')
plt.title('cluster vs genre')
plt.xticks(ind + width/2., ('C0','C1','C2','C4'))
plt.yticks(np.arange(0, 700, 100))
lgd=plt.legend( colnames,loc = 'center left', bbox_to_anchor = (1,0.5))

plt.savefig('clustergenre.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')