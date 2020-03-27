
"""
todo:

to remove ambiguity regarding patch_x,patch_y,window_x,window_y
"""


import numpy as np
from sklearn.decomposition import PCA
from stemclustering import stemClustering
from stemdescriptor import get_descriptor
from stempower_spectrum import get_power_spectrum
from sklearn.cluster import KMeans

import gc

descriptors_implemented = ['power_spectrum','local_correlation_map']

class segmentationSTEM:
    def __init__(self,n_patterns=2, 
                 window_x=21,window_y=21,
                 descriptor_name='local_correlation_map',
                 n_PCA_components=5,
                 pca_fitted=None,kmeans_init_centers=None,
                 one_step_kmeans=False,
                 #parameters assicated with local_correlation_map descriptors
                 patch_x=20,patch_y=20, max_num_points=100, parallel=True,
                 #
                 ):
        
        """
        n_patterns.........number of periodic patterns which the image is segmented into
        patch_x............height of patch
        patch_y............width of patch
        window_x...........height of window
        window_y...........width of window
        descriptor_name....str, name of the descriptor to be used, should be in descriptors_implemented
        n_PCA_components...number of principle components used for segmentation
        one_step_kmeans....Boolean, if True, only run kmeans for one step.
        max_num_points.....the maximum number of points to be chosen uniformly 
                           from the local correlation map. 
                           The more points we use, the more accurate the results are,
                           but the more RAM memory is required, which may exceed the available memory.
                           This parameter is not exactly equal to the number of points that is actually used
                           because we use the uniform grid points.
        parallel...........True: use openmp in the calculation of descriptors
                           False: serial
        """
        self._PCA_components = []
        self._segmentation_labels = []
        if descriptor_name not in descriptors_implemented:
            raise ValueError('descriptor_name should be in ', descriptors_implemented)
        self.paras = {'n_patterns':n_patterns,
                      'patch_x':patch_x,
                      'patch_y':patch_y,
                      'window_x':window_x,
                      'window_y':window_y,
                      'descriptor_name':descriptor_name,
                      'pca_fitted':pca_fitted,
                      'kmeans_init_centers':kmeans_init_centers,
                      'one_step_kmeans':one_step_kmeans,
                      'n_PCA_components':n_PCA_components,
                      'max_num_points':max_num_points, 
                      'parallel':parallel}

    def get_PCA_components(self, image):
        self.check_image_validity(image)
        if self.paras['descriptor_name'] is 'local_correlation_map':
            descriptors = get_descriptor(image,self.paras['patch_x'],
                                        self.paras['patch_y'],
                                        self.paras['window_x'],
                                        self.paras['window_y'],
                                        self.paras['max_num_points'],
                                        parallel=self.paras['parallel'])
        elif self.paras['descriptor_name'] is 'power_spectrum':
            descriptors = get_power_spectrum(image,2*self.paras['window_x'],2*self.paras['window_y'])
        n_components = self.paras['n_PCA_components']
        shape = descriptors.shape
        if self.paras['pca_fitted'] is None:
            pca = PCA(n_components)
            self._PCA_components = np.reshape(pca.fit_transform(np.reshape(descriptors,(-1,shape[2]))), (shape[0],shape[1],n_components))
        else:
            pca = self.paras['pca_fitted']
            self._PCA_components = np.reshape(self.paras['pca_fitted'].transform(np.reshape(descriptors,(-1,shape[2]))), (shape[0],shape[1],n_components)) 
        self._pca = pca
        return self._PCA_components

    def perform_clustering(self, image):
        """
        image..............2D numpy array
        """
        features = self.get_PCA_components(image)
        shape = features.shape
        if self.paras['one_step_kmeans'] is True:
            max_iter = 1
        else:
            max_iter = 300
        if self.paras['kmeans_init_centers'] is None:
            kmeans = KMeans(n_clusters=self.paras['n_patterns'],max_iter=max_iter)
        else:
            kmeans = KMeans(n_clusters=self.paras['n_patterns'],max_iter=max_iter,init=self.paras['kmeans_init_centers'])
        kmeans.fit(np.reshape(features,(-1,shape[2])))
        self._kmeans = kmeans
        self._segmentation_labels = np.zeros_like(image,dtype=np.int32) - 1
        shape_image = image.shape
        window_x = self.paras['window_x']
        window_y = self.paras['window_y']
        patch_x = self.paras['patch_x']
        patch_y = self.paras['patch_y']
        if self.paras['descriptor_name'] is 'power_spectrum':
            self._segmentation_labels[window_x:(shape_image[0]-window_x),window_y:(shape_image[1]-window_y)] =\
                   np.reshape(kmeans.labels_, (shape[0], shape[1]))
        elif self.paras['descriptor_name'] is 'local_correlation_map':
            self._segmentation_labels[(window_x+patch_x):(shape_image[0]-window_x-patch_x),(window_y+patch_y):(shape_image[0]-window_y-patch_y)] =\
                   np.reshape(kmeans.labels_, (shape[0], shape[1]))
        return self._segmentation_labels
    
    def check_image_validity(self,image):
        if not type(image).__module__ == np.__name__:
            raise ValueError("image should be 2D numpy array")
        elif not image.ndim == 2:
            raise ValueError("image should be 2D numpy array")


