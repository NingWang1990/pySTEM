import numpy as np
from sklearn.decomposition import PCA
from stemclustering import stemClustering
from stemdescriptor import get_descriptor

class segmentationSTEM:
    def __init__(self,n_patterns=2, patch_x=20,patch_y=20,window_x=21,window_y=21,
                 n_PCA_components=5, max_num_points=100, parallel=True):
        
        """
        n_patterns.........number of periodic patterns which the image is segmented into
        patch_x............height of patch
        patch_y............width of patch
        window_x...........height of window
        window_y...........width of window
        n_PCA_components...number of principle components used for segmentation
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
        self.paras = {'n_patterns':n_patterns,
                      'patch_x':patch_x,
                      'patch_y':patch_y,
                      'window_x':window_x,
                      'window_y':window_y,
                      'n_PCA_components':n_PCA_components,
                      'max_num_points':max_num_points, 
                      'parallel':parallel}

    def get_PCA_components(self, image):
        self.check_image_validity(image)
        descriptors = get_descriptor(image,self.paras['patch_x'],
                                      self.paras['patch_y'],
                                      self.paras['window_x'],
                                      self.paras['window_y'],
                                      self.paras['max_num_points'],
                                      parallel=self.paras['parallel'])
        n_components = self.paras['n_PCA_components']
        pca = PCA(n_components)
        shape = descriptors.shape
        self._PCA_components = np.reshape(pca.fit_transform(np.reshape(descriptors,(-1,shape[2]))), (shape[0],shape[1],n_components))
        del descriptors
        return self._PCA_components

    def perform_clustering(self, image):
        """
        image..............2D numpy array
        """
        features = self.get_PCA_components(image)
        self._segmentation_labels = stemClustering(features,method='Kmean',n_clusters=self.paras['n_patterns']) 
        return self._segmentation_labels
    
    def check_image_validity(self,image):
        if not type(image).__module__ == np.__name__:
            raise ValueError("image should be 2D numpy array")
        elif not image.ndim == 2:
            raise ValueError("image should be 2D numpy array")


