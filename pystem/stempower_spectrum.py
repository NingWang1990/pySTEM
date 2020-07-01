"""
a quick and dirty implementation of descriptors based on power spectrum
"""


import numpy as np
from numpy.lib import stride_tricks as st

def get_power_spectrum_m1(image, window_x,window_y,step=5,logarithm=True):
    shape = image.shape
    x_index = np.arange(window_x, shape[0]-window_x,step)
    y_index = np.arange(window_y, shape[1]-window_y,step)
    descriptors = np.zeros((len(x_index),len(y_index),2*window_x+1,2*window_y+1),dtype=np.float32)
    wx = np.hanning(2*window_x+1)
    wy = np.hanning(2*window_y+1)
    hanning2d = np.sqrt(np.outer(wx, wy))
    for i, ii in enumerate(x_index):
        for j, jj in enumerate(y_index):
            img_window = image[(ii-window_x):(ii+window_x+1), (jj-window_y):(jj+window_y+1)].copy()
            img_window = img_window - np.mean(img_window)
            #img_window /= np.std(img_window)
            img_window = img_window*hanning2d
            if logarithm == True:
                descriptors[i,j] = np.log(np.abs(np.fft.fft2(img_window))**2 + 1.)
            else:
                descriptors[i,j] = np.abs(np.fft.fft2(img_window))**2
    shape_descriptors = descriptors.shape
    return np.reshape(descriptors,(shape_descriptors[0], shape_descriptors[1],-1))

def mf(A, k_shape= (3, 3),step=1):
    m= int( (A.shape[0]- k_shape[0]+1)/step)
    n= int( (A.shape[1]- k_shape[1]+1)/step)
    #strides = A.strides+ A.strides
    strides = (A.strides[0]*step, A.strides[1]*step,A.strides[0],A.strides[1])
    new_shape= (m, n, k_shape[0], k_shape[1])
    A= st.as_strided(A, shape= new_shape, strides= strides)
    return A


def get_power_spectrum_m2(image,window_x,window_y,step=5, logarithm=True):
    # to include edge
    window_x = 2*window_x + 1
    window_y = 2*window_y + 1
    imageW = mf(image,k_shape=(window_x, window_y),step=step)
    imageW = imageW - np.mean(imageW,axis=(2,3),keepdims=True)
    #imageW = imageW / np.std(imageW,axis=(2,3),keepdims=True)
    wx = np.hanning(window_x)
    wy = np.hanning(window_y)
    w2d = np.reshape(np.sqrt(np.outer(wx, wy)),(1,1,window_x,window_y))
    imageW = imageW*w2d
    #descriptors = np.abs(np.fft.fft2(imageW))**2/(window_x*window_y)
    descriptors = np.abs(np.fft.fft2(imageW))**2
    if logarithm == True:
        descriptors = np.log(descriptors+1.)
    shape = descriptors.shape
    descriptors = np.reshape(descriptors, (shape[0],shape[1],-1))
    return descriptors
