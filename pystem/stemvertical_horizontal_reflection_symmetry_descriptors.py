import numpy as np

def get_reflection_symmetry_descriptors(image,window_x=20, window_y=20,patch_x=20,patch_y=20, step_symmetry_analysis=5,num_max=10):
    image_symmetry = image_reflection_symmetry(image,patch_x=patch_x,patch_y=patch_y,step=step_symmetry_analysis)
    shape = image.shape
    x_index = np.arange(window_x+patch_x, shape[0]-window_x-patch_x,step_symmetry_analysis)
    y_index = np.arange(window_y+patch_y, shape[1]-window_y-patch_y,step_symmetry_analysis)
    descriptors = np.zeros((len(x_index),len(y_index),num_max,2),dtype=np.float32)
    if (2*window_x+1)*(2*window_y+1)<num_max:
        num_max = (2*window_x+1)*(2*window_y+1)
    for i,ii in enumerate(x_index):
        for j,jj in enumerate(y_index):
            des_window = image_symmetry[(ii-window_x):(ii+window_x+1),(jj-window_y):(jj+window_y+1),:].copy()
            des_window = np.reshape(des_window, (-1,2))
            
            des_window = -np.sort(-des_window, axis=0)
            descriptors[i,j,:num_max,:] = des_window[:num_max,:]
    
    return np.reshape(descriptors,(len(x_index),len(y_index),num_max*2))
            #descriptors[i-radius-window_x,j-radius-window_y,0] = two_fold_symmetry

def image_reflection_symmetry(image, patch_x=20, patch_y=20, step=1):
    shape = image.shape
    image_symmetry = np.zeros(shape+(2,),dtype=np.float32) - 1.
    for i in range(patch_x, shape[0]-patch_x, step):
        for j in range(patch_y, shape[1]-patch_y, step):
            image_patch = image[(i-patch_x):(i+patch_x+1),(j-patch_y):(j+patch_y+1)].copy()
            image_patch = image_patch - np.mean(image_patch)
            image_symmetry[i,j,0] = cross_autocorrelation(image_patch, np.flip(image_patch,axis=0))
            image_symmetry[i,j,1] = cross_autocorrelation(image_patch, np.flip(image_patch,axis=1))
    return image_symmetry


def cross_autocorrelation(a,b):
    return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

