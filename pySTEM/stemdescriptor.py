import numpy as np
import time
import os
import _stemdescriptor



def get_num_shifts(window_x,window_y,grid):
    
    n_x = int(2*window_x/grid) + 1
    n_y = int(2*window_y/grid) + 1
    return n_x*n_y


def get_descriptor(image, patch_x=11,patch_y=5,window_x=51,window_y=51,num_points=100,step=3,parallel=True,removing_mean=True):

    if num_points <=2:
        raise ValueError('increase num_points')
    start = time.time()
    image = image.astype(np.float32)
    num_rows = len(image)
    num_cols = len(image[0])
    count = 0
     
    grid = 1
    num_shifts = get_num_shifts(window_x,window_y,grid)
    while num_shifts > num_points:
        grid += 1
        num_shifts = get_num_shifts(window_x,window_y,grid)
    n_descriptors = num_shifts
    shape = image.shape
    x_index = np.arange(window_x+patch_x, shape[0]-window_x-patch_x, step)
    y_index = np.arange(window_y+patch_y, shape[1]-window_y-patch_y, step)
    descriptors = np.zeros((len(x_index),len(y_index),n_descriptors),dtype=np.float32)
    num_rows_desp = len(descriptors)
    num_cols_desp = len(descriptors[0])
    descriptors = descriptors.flatten()
    _stemdescriptor.calc(np.reshape(image,-1),descriptors,num_rows,num_cols,patch_x,patch_y,
                          window_x,window_y,grid,grid,n_descriptors,step,num_rows_desp,num_cols_desp,int(removing_mean))
    return np.reshape(descriptors, (len(x_index), len(y_index), n_descriptors))
