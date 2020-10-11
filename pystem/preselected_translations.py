import numpy as np
import _preselectedtranslations
def preselected_translations(image, translations,patch_x=11,patch_y=5, step=3,removing_mean=True):

    image = image.astype(np.float32)
    translations = translations.astype(np.int32)
    window_x = np.max(np.abs(translations[:,0]))
    window_y = np.max(np.abs(translations[:,1]))
    n_descriptors = len(translations)
    num_rows = len(image)
    num_cols = len(image[0])
    shape = image.shape
    x_index = np.arange(window_x+patch_x, shape[0]-window_x-patch_x, step)
    y_index = np.arange(window_y+patch_y, shape[1]-window_y-patch_y, step)
    descriptors = np.zeros((len(x_index),len(y_index),n_descriptors),dtype=np.float32)
    num_rows_desp = len(descriptors)
    num_cols_desp = len(descriptors[0])
    descriptors = descriptors.flatten()

    # compute descriptor
    
    _preselectedtranslations.calc(image.flatten(),descriptors,translations.flatten(),
                                  num_rows,num_cols,patch_x,patch_y,window_x,window_y,n_descriptors,step,
                                  num_rows_desp, num_cols_desp, removing_mean)
    return np.reshape(descriptors, (len(x_index), len(y_index), n_descriptors))
