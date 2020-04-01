
"""
todo:

to remove ambiguity regarding patch_x,patch_y,window_x,window_y
"""


import numpy as np
from sklearn.decomposition import PCA
from stemclustering import stemClustering
from stemdescriptor import get_descriptor
from stempower_spectrum import get_power_spectrum_m1, get_power_spectrum_m2
from stemrotational_symmetry_descriptors import get_rotational_symmetry_descriptors
from stemreflection_symmetry_descriptors import get_reflection_symmetry_descriptors
from sklearn.cluster import KMeans
from scipy.ndimage import map_coordinates
import gc

descriptors_implemented = ['power_spectrum','local_correlation_map','rotational_symmetry_maximums','reflection_symmetry_maximums']

class segmentationSTEM:
    def __init__(self,n_patterns=2, 
                 window_x=21,window_y=21,
                 descriptor_name='local_correlation_map',
                 n_PCA_components=5,
                 upsampling=True,
                 pca_fitted=None,kmeans_init_centers=None,
                 one_step_kmeans=False,
                 # 
                 num_reflection_plane=10,
                 # parameters associated with rotational with rotational_symmetry_maximums descriptors
                 radius=20,nr=20,nt=60,step=5,num_max=10,
                 #parameters associated with local_correlation_map descriptors
                 patch_x=20,patch_y=20, max_num_points=100, parallel=True,
                 #
                 power_spectrum_logarithm=True,
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
                      'radius':radius,
                      'nr':nr,
                      'nt':nt,
                      'step':step,
                      'num_max': num_max,
                      'upsampling':upsampling,
                      'num_reflection_plane': num_reflection_plane,
                      'pca_fitted':pca_fitted,
                      'kmeans_init_centers':kmeans_init_centers,
                      'one_step_kmeans':one_step_kmeans,
                      'n_PCA_components':n_PCA_components,
                      'max_num_points':max_num_points, 
                      'power_spectrum_logarithm':power_spectrum_logarithm,
                      'parallel':parallel}

    def get_PCA_components(self, image):
        self.check_image_validity(image)
        if self.paras['descriptor_name'] is 'local_correlation_map':
            descriptors = get_descriptor(image,self.paras['patch_x'],
                                        self.paras['patch_y'],
                                        self.paras['window_x'],
                                        self.paras['window_y'],
                                        self.paras['max_num_points'],
                                        step = self.paras['step'],
                                        parallel=self.paras['parallel'])
        elif self.paras['descriptor_name'] is 'power_spectrum':
            descriptors = get_power_spectrum_m1(image,2*self.paras['window_x'],2*self.paras['window_y'], step=self.paras['step'],logarithm=self.paras['power_spectrum_logarithm'])
        elif self.paras['descriptor_name'] is 'rotational_symmetry_maximums':
            descriptors = get_rotational_symmetry_descriptors(image, window_x=self.paras['window_x'], window_y=self.paras['window_y'],
                                                              radius=self.paras['radius'],nr=self.paras['nr'], nt=self.paras['nt'],
                                                              num_max=self.paras['num_max'],step_symmetry_analysis=self.paras['step'])
        elif self.paras['descriptor_name'] is 'reflection_symmetry_maximums':
            descriptors = get_reflection_symmetry_descriptors(image,window_x=self.paras['window_x'],window_y=self.paras['window_y'],
                                                              radius=self.paras['radius'],nr=self.paras['nr'], nt=self.paras['nt'],
                                                              num_reflection_plane=self.paras['num_reflection_plane'],
                                                              step_symmetry_analysis=self.paras['step'],num_max=self.paras['num_max'])
            
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


    def perform_upsampling_labels(self,image, labels):
        
        step = self.paras['step']
        shape_image = image.shape
        shape_labels = labels.shape
        x_index = np.arange(step*shape_labels[0])
        diff = shape_image[0] - len(x_index)
        diff_left = int( diff / 2.)
        diff_right = diff - diff_left
        left_index = np.flip(-np.arange(1, 1+diff_left))
        right_index = len(x_index)+np.arange(diff_right)
        x_index = np.concatenate((left_index, x_index, right_index))

        y_index = np.arange(step*shape_labels[1])
        diff = shape_image[1] - len(y_index)
        diff_left = int( diff / 2.)
        diff_right = diff - diff_left
        left_index = np.flip(-np.arange(1, 1+diff_left))
        right_index = len(y_index)+np.arange(diff_right)
        y_index = np.concatenate((left_index, y_index, right_index))
            
        x_grid, y_grid = np.meshgrid(x_index, y_index)
        x_grid = (x_grid / step).flatten()
        y_grid = (y_grid / step).flatten()
        coords = np.vstack((x_grid, y_grid))
        labels_up = map_coordinates(labels, coords,mode='nearest')
        labels_up = np.reshape(labels_up, (shape_image[1], shape_image[0])).T
        labels_up = np.round(labels_up).astype(np.int32) % self.paras['n_patterns']
        return labels_up


    def perform_clustering(self, image):
        """
        image..............2D numpy array
        upsampling.........Boolean, if True, upsampling in order to match the shape of image.
        """
        image = image.astype(np.float32)
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
        labels = np.reshape(kmeans.labels_, (shape[0],shape[1]))
        self._segmentation_labels = np.zeros_like(image,dtype=np.int32) - 1
        shape_image = image.shape
        window_x = self.paras['window_x']
        window_y = self.paras['window_y']
        patch_x = self.paras['patch_x']
        patch_y = self.paras['patch_y']
        if self.paras['upsampling'] is True:
            return self.perform_upsampling_labels(image=image, labels=labels)
        else:
            return labels
        
        #if self.paras['descriptor_name'] is 'power_spectrum':
        #    self._segmentation_labels[window_x:(shape_image[0]-window_x),window_y:(shape_image[1]-window_y)] =\
        #           np.reshape(kmeans.labels_, (shape[0], shape[1]))
        #elif self.paras['descriptor_name'] is 'rotational_symmetry_maximums':
        #    self._segmentation_labels = np.reshape(kmeans.labels_, (shape[0], shape[1]))
        #elif self.paras['descriptor_name'] is 'reflection_symmetry_maximums':
        #    self._segmentation_labels = np.reshape(kmeans.labels_, (shape[0], shape[1]))
        #elif self.paras['descriptor_name'] is 'local_correlation_map':
        #    self._segmentation_labels[(window_x+patch_x):(shape_image[0]-window_x-patch_x),(window_y+patch_y):(shape_image[0]-window_y-patch_y)] =\
        #           np.reshape(kmeans.labels_, (shape[0], shape[1]))
        #return self._segmentation_labels
    
    def check_image_validity(self,image):
        if not type(image).__module__ == np.__name__:
            raise ValueError("image should be 2D numpy array")
        elif not image.ndim == 2:
            raise ValueError("image should be 2D numpy array")


