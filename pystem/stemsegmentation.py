
"""
todo:

1. store reflection planes and rotation axes 
2. to remove ambiguity regarding patch_x,patch_y,window_x,window_y
   extend soft_segmentation to n_patterns > 2 cases.
3. store descriptors, and avoid double calculating the same image
4. move underscore of the attributes to the end 
5. None appears in Fisher separability if 'fft' method is chosen
6. in upsampling, use patch_x and window_x to estimate left margin.
"""


import numpy as np
from sklearn.decomposition import PCA
from pystem.stemclustering import stemClustering
from pystem.stemdescriptor import get_descriptor
from pystem.preselected_translations import preselected_translations
from pystem.stempower_spectrum import get_power_spectrum_m1, get_power_spectrum_m2
from pystem.stemrotational_symmetry_descriptors import get_rotational_symmetry_descriptors
from pystem.stemreflection_symmetry_descriptors import get_reflection_symmetry_descriptors
from pystem.util import calculate_Fisher_separability
from sklearn.cluster import KMeans
from scipy.ndimage import map_coordinates
from scipy.interpolate import NearestNDInterpolator
import gc
import warnings

descriptors_implemented = ['power_spectrum','local_correlation_map',
                           'preselected_translations',
                           'rotational_symmetry_maximum_pooling',
                           'reflection_symmetry_maximum_pooling']
methods_implemented = ['direct','fft']
class segmentationSTEM:
    def __init__(self,n_patterns=2, stride=5,
                 descriptor_name='local_correlation_map',
                 n_PCA_components=5,
                 upsampling=True,
                 preselected_translations=None,
                 window_x=21,window_y=21, # required for descriptors other than preselected_translations
                 num_reflection_plane=10,
                 radius=20,
                 patch_x=20,patch_y=20, 
                 max_num_points=100,
                 method = 'direct',
                 sort_labels_by_pattern_size=True,
                 random_state=None,
                 # separability analysis
                 separability_analysis=False,
                 num_operations_with_best_sep=5,
                 one_step_kmeans=False,
                 # paras for pre-defined PCA and kmeans 
                 pca_fitted=None,kmeans_init_centers=None,

                 ):
        
        """
        n_patterns......................number of periodic patterns which the image is segmented into
        stride..........................int, the stride size for both vertical and horizontal directions 
        descriptor_name.................str, name of the descriptor to be used, should be in descriptors_implemented
        patch_x.........................half height of patch
        patch_y.........................half width of patch
        window_x........................half height of window
        window_y........................half width of window
        n_PCA_components................number of principle components used for segmentation
        upsampling......................Boolean, if True, perform upsampling to make the output labels have the same shape
                                                 with the input image
        sort_labels_by_pattern_size.....Boolean. If True, sort the pattern labels by size of patterns.
                                                 The smallest pattern has a label of 0, and the largest pattern has a label of n_patterns-1.
                                        This hyperparameter is added because Kmeans starts with random initialization. In different runs, 
                                        we may get different labels for the same pattern if this hyperparameter is set to False. 
        random_state....................int or None. It determines the random number generation for centroid initialization 
                                        in Kmeans clustering. Default is None. 
        separability_analysis...........Boolean, If True, calculate Fisher's separability for every feature.
        num_operations_with_best_sep....int, number of symmetry operations with best separability to select after seperability analysis
        preselected_translations........2D numpy array, (n_translations, 2).
        one_step_kmeans.................Boolean, if True, only run kmeans for one step.
        max_num_points..................the maximum number of points to be chosen uniformly 
                                        from the local correlation map. 
                                        The more points we use, the more accurate the results are,
                                        but the more RAM memory is required, which may exceed the available memory.
                                        This parameter is not exactly equal to the number of points that is actually used
                                        because we use the uniform grid points.
        """
        self._PCA_components = []
        self._segmentation_labels = []
        if descriptor_name not in descriptors_implemented:
            raise ValueError('descriptor_name should be in ', descriptors_implemented)
        if method not in methods_implemented:
            raise ValueError('method should be in', methods_implemented)
        if descriptor_name == 'preselected_translations':
            window_x = np.max(np.abs(preselected_translations[:,0]))
            window_y = np.max(np.abs(preselected_translations[:,1]))

        self.paras = {'n_patterns':n_patterns,
                      'patch_x':patch_x,
                      'patch_y':patch_y,
                      'window_x':window_x,
                      'window_y':window_y,
                      'descriptor_name':descriptor_name,
                      'separability_analysis':separability_analysis,
                      'num_operations_with_best_sep':num_operations_with_best_sep,
                      'preselected_translations':preselected_translations,
                      'removing_mean': True,
                      'radius':radius,
                      #'nr':nr,
                      #'nt':nt,
                      'step':stride,
                      'num_max': 10,
                      'upsampling':upsampling,
                      'random_state':random_state, 
                      'sort_labels_by_pattern_size':sort_labels_by_pattern_size,
                      'num_reflection_plane': num_reflection_plane,
                      'pca_fitted':pca_fitted,
                      'kmeans_init_centers':kmeans_init_centers,
                      'one_step_kmeans':one_step_kmeans,
                      'n_PCA_components':n_PCA_components,
                      'max_num_points':max_num_points, 
                      'power_spectrum_logarithm':True,
                      'soft_segmentation':False,
                      'method':method,
                      'verbose':0}

    
    def get_descriptors(self,image):    
        if (self.paras['verbose']>1): print ("Checking image validity", flush=True)
        if (self.paras['verbose']>0): print ("Computing descriptors via ", self.paras['descriptor_name'], flush=True)
        self.check_image_validity(image)
        if self.paras['descriptor_name'] == 'local_correlation_map':
            self._descriptors, self._translation_vectors = get_descriptor(image,self.paras['patch_x'],
                                                                self.paras['patch_y'],
                                                                self.paras['window_x'],
                                                                self.paras['window_y'],
                                                                self.paras['max_num_points'],
                                                                step = self.paras['step'],
                                                                method=self.paras['method'],
                                                                removing_mean=self.paras['removing_mean'])
        elif self.paras['descriptor_name'] == 'preselected_translations':
            self._descriptors = preselected_translations(image,self.paras['preselected_translations'],
                                                         patch_x=self.paras['patch_x'],patch_y=self.paras['patch_y'],
                                                         step=self.paras['step'], removing_mean=self.paras['removing_mean'])

        elif self.paras['descriptor_name'] == 'power_spectrum':
            self._descriptors = get_power_spectrum_m1(image,self.paras['window_x'],
                                                                self.paras['window_y'], 
                                                                step=self.paras['step'],
                                                                logarithm=self.paras['power_spectrum_logarithm'])
        elif self.paras['descriptor_name'] == 'rotational_symmetry_maximum_pooling':
            self._descriptors = get_rotational_symmetry_descriptors(image, window_x=self.paras['window_x'], window_y=self.paras['window_y'],
                                                              radius=self.paras['radius'],
                                                              num_max=self.paras['num_max'],step_symmetry_analysis=self.paras['step'])
        elif self.paras['descriptor_name'] == 'reflection_symmetry_maximum_pooling':
            self._descriptors = get_reflection_symmetry_descriptors(image,window_x=self.paras['window_x'],window_y=self.paras['window_y'],
                                                              radius=self.paras['radius'],
                                                              num_reflection_plane=self.paras['num_reflection_plane'],
                                                              step_symmetry_analysis=self.paras['step'],num_max=self.paras['num_max'])
        return self._descriptors
    
    def get_PCA_components(self, image, descriptors=None):
        if descriptors is None:
            descriptors = self.get_descriptors(image)
        elif (self.paras['verbose']>0):
            print ("Using descriptors from argument...")
        if (self.paras['verbose']>0): print ("Perform PCA")
        n_components = self.paras['n_PCA_components']
        shape = descriptors.shape
        if n_components >= shape[-1]:
            warnings.warn('skip PCA because # of features not larger than PCA components')
            self._pca = None
            self._PCA_components = descriptors
            return self._PCA_components
        if self.paras['pca_fitted'] is None:
            pca = PCA(n_components)
            if (self.paras['verbose']>1):
               print (shape[0]*shape[1], "x", shape[2], "->", shape[0]*shape[1], "x", n_components)

            self._PCA_components = np.reshape(pca.fit_transform(np.reshape(descriptors,(-1,shape[2]))), (shape[0],shape[1],n_components))
        else:
            pca = self.paras['pca_fitted']
            if (self.paras['verbose']>1):
               print ("Using pca_fitted: ", np.shape(pca))
            self._PCA_components = np.reshape(self.paras['pca_fitted'].transform(np.reshape(descriptors,(-1,shape[2]))), (shape[0],shape[1],n_components)) 
        self._pca = pca
        return self._PCA_components


    def perform_upsampling_labels(self,image, labels):
        if (self.paras['verbose']>0): print ("Perform upsampling", flush=True)
        
        step = self.paras['step']
        shape_image = image.shape
        shape_labels = labels.shape
        
        x_index = np.arange(shape_image[0]) - self.paras['window_x'] - self.paras['patch_x']
        y_index = np.arange(shape_image[1]) - self.paras['window_y'] - self.paras['patch_y']
        x_grid, y_grid = np.meshgrid(x_index, y_index,indexing='ij')
        x_grid = (x_grid / step).flatten()
        y_grid = (y_grid / step).flatten()
        coords = np.vstack((x_grid, y_grid)).T
        if self.paras['soft_segmentation'] is False:
            x_labels, y_labels = np.meshgrid(np.arange(shape_labels[0]),np.arange(shape_labels[1]),indexing='ij')
            x_labels = x_labels.flatten()
            y_labels = y_labels.flatten()
            input_coords = np.vstack([x_labels, y_labels]).T
            interpolator = NearestNDInterpolator(input_coords, labels.flatten())
            labels_up = interpolator(coords)
            labels_up = np.reshape(labels_up, shape_image)
            #labels_up = np.round(labels_up).astype(np.int32) % self.paras['n_patterns']
        else:
            labels_up = map_coordinates(labels, coords,mode='nearest')
            labels_up = np.reshape(labels_up, (shape_image[0], shape_image[1]))
            labels_up = np.clip(labels_up,0.,self.paras['n_patterns']-1.)
        
        if self.paras['sort_labels_by_pattern_size'] is True:
            labels_up = self.sort_labels_by_pattern_size(labels_up)  

        return labels_up

    def sort_labels_by_pattern_size(self, labels):
        """
        labels...........2D int ndarray
        """
        if (self.paras['verbose']>0): print ("Sorting labels by size...", flush=True)
        shape = labels.shape
        labels = labels.flatten()
        labels_new = np.zeros_like(labels, dtype=np.int32)
        values = np.sort(np.unique(labels))
        sizes = np.zeros(len(values))
        for i, value in enumerate(values):
            sizes[i] = np.sum(labels==value)
        ordered_indices = np.argsort(sizes)
        values = values[ordered_indices]
        for i, value in enumerate(values):
            indices = np.where(labels==value)
            labels_new[indices] = i
        labels_new = np.reshape(labels_new, shape)
        # also need to reorder cluster centers
        self._kmeans.cluster_centers_ = self._kmeans.cluster_centers_[ordered_indices]
        return labels_new
            

    def perform_soft_segmentation(self, cluster_centers,features):
        """
        cluster_centers..........ndarray of shape(n_clusters, n_features)
        features.................ndarray of shape(n_samples, n_features)
        """
        if (self.paras['verbose']>0): print ("Soft segmentation", flush=True)
        import scipy
        (n_clusters, n_features) = cluster_centers.shape
        (n_samples, n_features_t) = features.shape
        if not (n_clusters == 2):
            raise ValueError('currently only implemented for the two-clusters case')
        if not (n_features == n_features_t):
            raise ValueError('# of features not equal')
        distances = np.zeros((n_samples, n_clusters))
        for i in range(n_clusters):
            distances[:,i] = np.linalg.norm(features - cluster_centers[i:i+1,:],axis=1)
        return scipy.special.softmax(distances,axis=1)[:,0]


    def perform_clustering(self, image, features=None, descriptors=None):
        """
        image..............2D numpy array
        features..............if given, 3D numpy array (grid_x,grid_y,n_PCA_components)
                              if not given, compute from descriptors
        descriptors...........if given, 3D numpy array (grid_x,grid_y,n_descriptors)
                              if not given (and neither are features), compute from image
        """
        if (features is None):
           # compute features via PCA
           image = image.astype(np.float32)
           features = self.get_PCA_components(image, descriptors)
        elif features.ndim != 3:
           raise ValueError('features must be 3D (grid_x,grid_y,n_PCA_components)')
        else:
           if (self.paras['verbose']>1): print ("Using dimension-reduced features from argument", flush=True)


        shape = features.shape
        if (self.paras['verbose']>0): print ("Kmeans clustering", flush=True)
        if self.paras['one_step_kmeans'] is True:
            max_iter = 1
        else:
            max_iter = 300
        if self.paras['kmeans_init_centers'] is None:
            kmeans = KMeans(n_clusters=self.paras['n_patterns'],max_iter=max_iter,
                            random_state=self.paras['random_state'])
        else:
            kmeans = KMeans(n_clusters=self.paras['n_patterns'],max_iter=max_iter,
                            random_state=self.paras['random_state'],init=self.paras['kmeans_init_centers'])
        kmeans.fit(np.reshape(features,(-1,shape[2])))
        self._kmeans = kmeans
        if self.paras['soft_segmentation'] is False:
            labels = np.reshape(kmeans.labels_, (shape[0],shape[1]))
        else:
            soft_labels = self.perform_soft_segmentation(kmeans.cluster_centers_, np.reshape(features,(-1,shape[2])))
            labels = np.reshape(soft_labels, (shape[0],shape[1]))
        
        # separability analysis
        if self.paras['separability_analysis'] is True:
            if not self.paras['descriptor_name'] == 'local_correlation_map':
                raise NotImplementedError('separability only implemented for local_correlation_map descriptor')
            features = np.reshape(self._descriptors, (-1, self._descriptors.shape[-1]))
            self._Fisher_separability = calculate_Fisher_separability(features, labels.flatten())
            ordered_indices = np.argsort(-self._Fisher_separability)
            self._operations_with_best_sep = self._translation_vectors[ordered_indices[:self.paras['num_operations_with_best_sep']]]

        # this line looks useless
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
        #elif self.paras['descriptor_name'] is 'rotational_symmetry_maximum_pooling':
        #    self._segmentation_labels = np.reshape(kmeans.labels_, (shape[0], shape[1]))
        #elif self.paras['descriptor_name'] is 'reflection_symmetry_maximum_pooling':
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



