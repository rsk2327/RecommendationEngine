
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from scipy.sparse import csr_matrix
import nimfa as nf


user_colnames=['userid','movieid','rating','timestamp']
data=pd.read_table(r'C:\Anaconda2\data\ml-100k\u.data',header=None,names=user_colnames)
#print data.head()

item_colnames=['movieid','movietitle','releasedate','videorelesaedate','IMDB URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime',
                'Documentary','Drame','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-fi','Thriller','War','Western']
item=pd.read_table(r'C:\Anaconda2\data\ml-100k\u.item',sep='|',header=None,names=item_colnames)
#print item.head()

movie_titles=item.movietitle.tolist()


user_movies=data.merge(item, on='movieid')
#print user_movies.head()

del user_movies['releasedate']
del user_movies['videorelesaedate']
del user_movies['IMDB URL']
del user_movies['unknown']

pivot = user_movies.pivot_table('rating','userid','movieid')
#print pivot.head()

pivot = pivot.fillna(0)
#print pivot.head()

Q=pivot.values

m, n = Q.shape

#print Q
W = Q>0.5
W[W==True] = 1
W[W==False] = 0
W = W.astype(np.float64, copy=False)

print Q.shape

sparseR = csr_matrix(Q)

als =  nf.Lsnmf(sparseR,seed="random_vcol",rank=15,max_iter=30,beta=0.5)

als_fit= als.factorize()


#%%
user_features = als_fit.basis()
movie_features = als_fit.coef()
predictedR = np.dot( user_features.todense() , movie_features.todense() )

#print np.max(predictedR)


def print_recommendations(W=W, Q=Q, Q_hat=predictedR, movie_titles=movie_titles):
    #Q_hat -= np.min(Q_hat)
    #Q_hat[Q_hat < 1] *= 5
    global predictedR
    predictedR -= np.min(predictedR)
    predictedR *= float(5) / np.max(predictedR)
    movie_ids = np.argmax(predictedR - 5 * W, axis=1)

    #print movie_ids
    for jj, movie_id in zip(range(m), movie_ids):
        #if Q_hat[jj, movie_id] < 0.1: continue
        print('User {} liked {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq > 3])))
        print('User {} did not like {}\n'.format(jj + 1, ', '.join([movie_titles[ii] for ii, qq in enumerate(Q[jj]) if qq < 3 and qq != 0])))
        print('\n User {} recommended movie is {} - with predicted rating: {}'.format(
                    jj + 1, movie_titles[movie_id], predictedR[jj, movie_id]))
        print('\n' + 100 *  '-' + '\n')
print_recommendations()




