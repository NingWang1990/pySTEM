import numpy as np
import time
import os
import _stemdescriptor



def get_num_shifts(window_x,window_y,grid):
    
    n_x = int(2*window_x/grid) + 1
    n_y = int(2*window_y/grid) + 1
    return n_x*n_y


def get_descriptor(image, patch_x=11,patch_y=5,window_x=51,window_y=51,num_points=100,parallel=True):

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
    descriptors = np.zeros(((num_rows-2*(patch_x+window_x))*(num_cols-2*(patch_y+window_y))*n_descriptors),np.float32)
    _stemdescriptor.calc(np.reshape(image,-1),descriptors,num_rows,num_cols,patch_x,patch_y,
                          window_x,window_y,grid,grid,n_descriptors )
    return np.reshape(descriptors, (num_rows-2*(patch_x+window_x),num_cols-2*(patch_y+window_y),n_descriptors))
