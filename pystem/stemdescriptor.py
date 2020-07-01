import numpy as np
import time
import os
import _stemdescriptor
import stemdescriptor2 as fftstem


def get_num_shifts(window_x,window_y,grid):
    
    n_x = int(2*window_x/grid) + 1
    n_y = int(2*window_y/grid) + 1
    return n_x*n_y


def good_fftsize(nn):

   """" Decide whether this is a good size for FFTs: nn=2^n 3^m 5^k 7^l """
   while (nn % 2 == 0):
      nn /= 2
   while (nn % 3 == 0):
      nn /= 3
   while (nn % 5 == 0):
      nn /= 5
   while (nn % 7 == 0):
      nn /= 7
   return (nn == 1)

def get_good_fftsize(minsize):

   while not good_fftsize(minsize):
      minsize += 1
   return minsize


methods_implemented = ['direct', 'fft']
def get_descriptor(image, patch_x=11,patch_y=5,window_x=51,window_y=51,num_points=100,step=3,parallel=True,removing_mean=True,method='direct'):
    print ('method:', method)
    if method not in methods_implemented:
        raise ValueError('method should be in ', methods_implemented)

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

    # --- initialize fft
    fftx = get_good_fftsize (2 * (window_x + patch_x) + 2)
    ffty = get_good_fftsize (2 * (window_y + patch_y) + 2)
    # note: fft has first dimension run fastest
    fft = fftstem.WindowFFT (ffty, fftx)
    # compute descriptor
    if method == 'direct':
        _stemdescriptor.calc(np.reshape(image,-1),descriptors,num_rows,num_cols,patch_x,patch_y,
                          window_x,window_y,grid,grid,n_descriptors,step,num_rows_desp,num_cols_desp,int(removing_mean))
    else:
        fft.calc(np.reshape(image,-1),descriptors,num_rows,num_cols,patch_x,patch_y,
                          window_x,window_y,grid,grid,n_descriptors,step,num_rows_desp,num_cols_desp)
    return np.reshape(descriptors, (len(x_index), len(y_index), n_descriptors))
