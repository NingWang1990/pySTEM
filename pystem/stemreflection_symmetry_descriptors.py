import numpy as np
from scipy.ndimage import map_coordinates
import heapq

def get_reflection_symmetry_descriptors(image,window_x=20, window_y=20,radius=20,nr=10,nt=60, num_reflection_plane=10,num_max=5, step_symmetry_analysis=5):
    
    # for convenience of symmetry analysis
    if not nt%4 == 0:
        nt = (int (nt/4) + 1) * 4
    
    image_symmetry = image_reflection_symmetry(image,radius=radius, nr=nr, nt=nt, step=step_symmetry_analysis,num_reflection_plane=num_reflection_plane)
    shape = image.shape
    x_index = np.arange(window_x+radius, shape[0]-window_x-radius,step_symmetry_analysis)
    y_index = np.arange(window_y+radius, shape[1]-window_y-radius,step_symmetry_analysis)
    descriptors = np.zeros((len(x_index),len(y_index),num_max,num_reflection_plane),dtype=np.float32)
    if (2*window_x+1)*(2*window_y+1)<num_max:
        num_max = (2*window_x+1)*(2*window_y+1)
    for i,ii in enumerate(x_index):
        for j,jj in enumerate(y_index):
            des_window = image_symmetry[(ii-window_x):(ii+window_x+1),(jj-window_y):(jj+window_y+1),:].copy()
            des_window = np.reshape(des_window, (-1,num_reflection_plane))
            
            # use sort to find n largest values   
            des_window = -np.sort(-des_window, axis=0)
            descriptors[i,j,:num_max,:] = des_window[:num_max,:]
    
    return np.reshape(descriptors,(len(x_index),len(y_index),num_max*num_reflection_plane))
            #descriptors[i-radius-window_x,j-radius-window_y,0] = two_fold_symmetry

def calculate_descriptor_for_specific_center(image, center=(100,100),window_x=20, window_y=20, radius=20,nr=10,nt=60,num_max=10):
    symmetry_map = np.zeros((2*window_x+1, 2*window_y+1,4))
    countx = 0
    for i in range(center[0]-window_x, center[0]+window_x+1):
        county = 0
        for j in range(center[1]-window_y, center[1]+window_y+1):
            origin = (i,j)
            polar_data, _,_ = reproject_image_into_polar(image, origin,radius=radius,nr=nr,nt=nt)
            symmetry_map[countx,county,0] = two_fold_symmetry(polar_data)
            symmetry_map[countx,county,1] = three_fold_symmetry(polar_data)
            symmetry_map[countx,county,2] = four_fold_symmetry(polar_data)
            symmetry_map[countx,county,3] = six_fold_symmetry(polar_data)
            county += 1
        countx += 1
    symmetry_map = np.reshape(symmetry_map,(-1,4))
    symmetry_map = -np.sort(-symmetry_map, axis=0)
    shape = symmetry_map.shape
    if shape[0] > num_max:
        return symmetry_map[:num_max,:]
    else:
        return symmetry_map[:,:]


def image_reflection_symmetry(image, radius=20, nr=10, nt=60, step=1,num_reflection_plane=10):
    shape = image.shape
    image_symmetry = np.zeros(shape+(num_reflection_plane,),dtype=np.float32) - 1.
    interval = int(nt/(2*num_reflection_plane))
    if interval < 1:
        raise ValueError("meshgrid of radial coordinates not enough for %d reflection planes" % num_reflection_plane)
        reflection_planes = np.arange(int(nt/2))
    else:
        reflection_planes = np.arange(num_reflection_plane)*interval
    half_nt = int(nt/2)
    quarter_plane = int(nt/4) 
    for i in range(radius, shape[0]-radius, step):
        for j in range(radius, shape[1]-radius, step):
            polar_data,_,_ = reproject_image_into_polar(image,origin=(i,j),radius=radius,nr=nr,nt=nt)
            for k,plane in enumerate(reflection_planes):
                data_roll = np.roll(polar_data, -plane, axis=1)
                data_24 = data_roll[:, quarter_plane:(quarter_plane+half_nt+1)]
                theta_index = np.arange((quarter_plane+half_nt+1),(quarter_plane+half_nt + half_nt) ) % nt
                data_13 = data_roll[:,theta_index]
                data_24_flip = np.flip(data_24, axis=1)
                data_13_flip = np.flip(data_13,axis=1)
                image_symmetry[i,j,k] = cross_autocorrelation(np.concatenate([data_13,data_24],axis=1),np.concatenate([data_13_flip,data_24_flip],axis=1)) 
    return image_symmetry


def reproject_image_into_polar(data, origin=(10,10),radius=10, Jacobian=True,
                               nr=10, nt=60):
    """
    Reprojects a 2D numpy array (**data**) into a polar coordinate system,
    with the pole placed at **origin** and the angle measured clockwise from
    the upward direction. The resulting array has rows corresponding to the
    radial grid, and columns corresponding to the angular grid.

    Parameters
    ----------
    data : 2D float np.array
        the image array
    origin : tuple or None
        (row, column) coordinates of the image origin. If ``None``, the
        geometric center of the image is used.
    Jacobian : bool
        Include `r` intensity scaling in the coordinate transform.
        This should be included to account for the changing pixel size that
        occurs during the transform.
    nr:     int, number of 
    Returns
    -------
    output : 2D np.array
        the polar image (r, theta)
    r_grid : 2D np.array
        meshgrid of radial coordinates
    theta_grid : 2D np.array
        meshgrid of angular coordinates

    adapted from:
    https://stackoverflow.com/questions/3798333/image-information-along-a-polar-coordinate-system

    """
    data = data.astype(np.float32)
    imageW = data[(origin[0]-radius):(origin[0]+radius+1), (origin[1]-radius):(origin[1]+radius+1)].copy()
    imageW = imageW - np.mean(imageW)
    #imageW = imageW / np.std(imageW)
    
    #imageW =  np.abs(np.fft.fft2(imageW))**2
    #imageW =  np.fft.fftshift(np.abs(np.fft.fft2(imageW))**2)
    #imageW = np.fft.fftshift(np.real(np.fft.ifft2(imageW)))
    # Make a regular (in polar space) grid
    r_i = np.linspace(1., radius, nr, endpoint=False)
    theta_i = np.linspace(0, 2.*np.pi, nt, endpoint=False)
    theta_grid, r_grid = np.meshgrid(theta_i, r_i)
    
    # Convert the r and theta grids to Cartesian coordinates
    X, Y = polar2cart(r_grid, theta_grid)
    # then to a 2×n array of row and column indices for np.map_coordinates()
    rowi = (X + radius).flatten()
    coli = (Y + radius).flatten()
    coords = np.vstack((rowi, coli))
    zi = map_coordinates(imageW, coords)
    output = zi.reshape((nr, nt))
    
    if Jacobian:
        output = output * r_i[:, np.newaxis]
    #return imageW, r_grid, theta_grid
    return output, r_grid, theta_grid

def cart2polar(x, y):
    """
    Transform Cartesian coordinates to polar.

    Parameters
    ----------
    x, y : floats or arrays
        Cartesian coordinates

    Returns
    -------
    r, theta : floats or arrays
        Polar coordinates

    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x, y)  # θ referenced to vertical
    return r, theta


def polar2cart(r, theta):
    """
    Transform polar coordinates to Cartesian.

    Parameters
    -------
    r, theta : floats or arrays
        Polar coordinates

    Returns
    ----------
    x, y : floats or arrays
        Cartesian coordinates
    """
    y = r * np.cos(theta)   # θ referenced to vertical
    x = r * np.sin(theta)
    return x, y

def cross_autocorrelation(a,b):
    return np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b))

