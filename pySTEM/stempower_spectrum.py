"""
a quick and dirty implementation of descriptors based on power spectrum
"""
import numpy as np

from numpy.lib import stride_tricks as st
def mf(A, k_shape= (3, 3)):
    m= A.shape[0]- k_shape[0]+1
    n= A.shape[1]- k_shape[1]+1
    strides= A.strides+ A.strides
    new_shape= (m, n, k_shape[0], k_shape[1])
    A= st.as_strided(A, shape= new_shape, strides= strides)
    return A

def get_power_spectrum(image,window_x,window_y,logarithm=True):
    
    # to include edge
    window_x += 1
    window_y += 1
    imageW = mf(image,k_shape=(window_x, window_y))
    imageW = imageW - np.mean(imageW,axis=(2,3),keepdims=True)
    imageW = imageW - np.std(imageW,axis=(2,3),keepdims=True)
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
