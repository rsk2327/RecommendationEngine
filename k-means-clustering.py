import sys
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans

item_colnames=['movieid','movietitle','releasedate','videorelesaedate','IMDB URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime',
                'Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-fi','Thriller','War','Western']
item=pd.read_table(r'C:\Anaconda2\data\ml-100k\u.item',sep='|',header=None,names=item_colnames)

del item['movieid']
del item['movietitle']
del item['releasedate']
del item['videorelesaedate']
del item['IMDB URL']
del item['unknown']

item = item.as_matrix()


kmeans = KMeans(n_clusters=4)
kmeans.fit(item)

centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)
'''
plt.scatter(centroids[:,0], centroids[:,1], marker = "x", s=10, linewidths = 5, zorder = 10)

plt.show()
'''
