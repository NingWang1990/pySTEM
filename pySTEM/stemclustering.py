import numpy as np
import time


methods_implemented = ['max','Kmean','GaussianMixture']

"""
Todo:
    GMM doesn't work for more than two lattices
"""


def stemClustering(descriptors,method='max',n_clusters=None):
    """
    return labels for clustering
    
    Input
    --------
    descriptors:  MxNxP numpy array
                  (M,N): the shape of the image
                  P:  number of features
    Output: labels
    """
    if method not in methods_implemented:
        raise ValueError('not implemented')
    start = time.time()
    if n_clusters == None:
        n_clusters = descriptors.shape[2]
    print ('Starting to perform clustering')
    shape = descriptors.shape
    if method is 'max':
        # n_clusters is useless for this method
        labels = np.zeros((shape[0],shape[1]))
        for i in range(shape[2]):
            mask = (np.zeros((shape[0],shape[1])) == 0.)
            for j in range(shape[2]):
                mask = mask & (descriptors[:,:,i] >= descriptors[:,:,j])
            labels += mask * i
        labels = labels.astype(int)
    elif method is 'Kmean':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters).fit(np.reshape(descriptors,(-1,shape[2])))
        labels = np.reshape(kmeans.labels_, (shape[0], shape[1]))
    elif method is 'GaussianMixture':
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=n_clusters)
        labels = gmm.fit_predict(np.reshape(descriptors,(-1,shape[2])))
        labels = np.reshape(labels, (shape[0], shape[1]))

    print ('time cost for clustering: %8.2f [s]' % (time.time()-start))
    return labels
