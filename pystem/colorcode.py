import colorsys
import numpy as np
try:
   import numba
   speedup = numba.njit
   def typed_list(colors):
      typed_colors = numba.typed.List ()
      [typed_colors.append(x) for x in colors]
      return typed_colors
except ImportError:
   def speedup(func): return func
   def typed_list(colors): return colors
   from warnings import warn
   warn ("Couldn't import 'numba' to speed up colorcoding. 'colorcode' will work, but be very slow.")

"""
   Auxiliary module for color-coding images according to the segmentation.

   Functions:
   ----------
   colorcode_hue......Returns list of hue values for the colors
   colorcode..........Performs the colorcoding
   colorcode_legend...Returns an array suitable to be used as a legend
"""

newhsv = speedup (colorsys.hsv_to_rgb)

def colorcode_hue(n_label):
    """
    Returns a list of hue values for colorcoding.

    n_label...number of labels

    Until n_label=6, this is hardcoded. For higher values,
    this is generated systematically instead.

    """
    if (n_label<=6):
        return [0.03,0.6,0.2,0.333,0.5,0.833][:n_label]
    else:
        return [0.833/(n_label-1) * i for i in range(n_label)]

def colorcode_legend(labels,saturation=0.2):
   """
   Returns rgb legend for colorcoding as a numpy array (1,n_label,3).

   label........array with labels
   saturation...saturation value (in hsv model)
   """
   maxlabel=np.max(labels)+1
   res = np.zeros((1,maxlabel,3))
   ch = colorcode_hue(maxlabel)
   for x in range(maxlabel):
       res[0,x,:] = newhsv(ch[x],saturation,1)
   return res

def colorcode(image,labels,colors=None,saturation=0.2,im0=None,im1=None):
    """
    rgb colorcoded intensity map as numpy array (nx,ny,3) from grayscale image and labels.

    image........intensity (min..max mapped to 0..1 intensity)
    label........int32 array with labels, same dimensions as image
    colors.......list of colors as hue values (hsv model)
    saturation...saturation values (in hsv model)
    im0..........Minimum image value (if known)
    im1..........Maximum image value (if known)
    """
    if image.shape != labels.shape:
       raise(ValueError, 'labels and image have different shape')
    if labels.dtype != np.int32:
       raise(TypeError, 'labels must be numpy int32 arrays')
    if np.min(labels) < 0:
       raise(ValueError, 'labels must be non-negative')
    if (colors is None):
       colors = colorcode_hue(np.max(labels)+1)
    rgbimg = np.zeros(list(image.shape)+[3])
    if im0 is None:
       im0=np.min(image)
    if im1 is None:
       im1=np.max(image)
    im_mag = im1 - im0
    @speedup
    def innercolorcode (image,labels,rgbimg,colors):
        for (x,y) in np.ndindex(image.shape):
            #hue = offset+scale * labels[x,y]
            #if (hue < 0): hue += 1.
            #if (hue > 1): hue -= 1.
            hue=colors[labels[x,y]]
    #        rgbimg[x,y,:] = colorsys.hsv_to_rgb(hue,saturation,(image[x,y]-im0)/im_mag)
            rgbimg[x,y,:] = newhsv(hue,saturation,(image[x,y]-im0)/im_mag)
    innercolorcode (image,labels,rgbimg,typed_list(colors))
    return rgbimg
