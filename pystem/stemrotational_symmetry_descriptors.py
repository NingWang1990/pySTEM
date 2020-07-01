import numpy as np
from scipy.ndimage import map_coordinates
import heapq

def get_rotational_symmetry_descriptors(image,window_x=20, window_y=20,radius=20,nr=10,nt=60, num_max=10, step_symmetry_analysis=5):
    num_max = np.min([num_max, (2*window_x+1)*(2*window_y+1)])
    image_symmetry = image_rotational_symmetry(image,radius=radius, nr=nr, nt=nt, step=step_symmetry_analysis)
    shape = image.shape
    x_index = np.arange(window_x+radius, shape[0]-window_x-radius,step_symmetry_analysis)
    y_index = np.arange(window_y+radius, shape[1]-window_y-radius,step_symmetry_analysis)
    descriptors = np.zeros((len(x_index),len(y_index),num_max,4),dtype=np.float32)
    if (2*window_x+1)*(2*window_y+1)<num_max:
        num_max = (2*window_x+1)*(2*window_y+1)
    for i,ii in enumerate(x_index):
        for j,jj in enumerate(y_index):
            des_window = image_symmetry[(ii-window_x):(ii+window_x+1),(jj-window_y):(jj+window_y+1),:].copy()
            des_window = np.reshape(des_window, (-1,4))
            
            # use partition to find n largest values
            #des_window  = -np.partition(-des_window,num_max,axis=0)[:num_max,:]
            #descriptors[i,j,:num_max,:] = -np.sort(-des_window, axis=0)
            
            # use heapq to find n largest values
            #descriptors[i,j,:num_max,0] = heapq.nlargest(num_max, des_window[:,0])
            #descriptors[i,j,:num_max,1] = heapq.nlargest(num_max, des_window[:,1])
            #descriptors[i,j,:num_max,2] = heapq.nlargest(num_max, des_window[:,2])
            #descriptors[i,j,:num_max,3] = heapq.nlargest(num_max, des_window[:,3])

            # use sort to find n largest values   
            des_window = -np.sort(-des_window, axis=0)
            descriptors[i,j,:num_max,:] = des_window[:num_max,:]
    
    return np.reshape(descriptors,(len(x_index),len(y_index),num_max*4))
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


def image_rotational_symmetry(image, radius=20, nr=10, nt=60, step=1):
    shape = image.shape
    image_symmetry = np.zeros(shape+(4,),dtype=np.float32) - 1.
    for i in range(radius, shape[0]-radius, step):
        for j in range(radius, shape[1]-radius, step):
            polar_data,_,_ = reproject_image_into_polar(image,origin=(i,j),radius=radius,nr=nr,nt=nt)
            image_symmetry[i,j,0] = two_fold_symmetry(polar_data)
            image_symmetry[i,j,1] = three_fold_symmetry(polar_data)
            image_symmetry[i,j,2] = four_fold_symmetry(polar_data)
            image_symmetry[i,j,3] = six_fold_symmetry(polar_data)
    return image_symmetry


def two_fold_symmetry(polar_data):
    """
    polar_data: image array in the polar coordinate system, (r, theta)
    """
    
    shape = polar_data.shape
    shift = int(shape[1] / 2)
    data_rolled = np.roll(polar_data,shift, axis=1)
    return cross_autocorrelation(polar_data, data_rolled)

def three_fold_symmetry(polar_data):
    """
    polar_data: image array in the polar coordinate system, (r, theta)
    """
    
    shape = polar_data.shape
    shift = int(shape[1] / 3)
    data_rolled = np.roll(polar_data,shift, axis=1)
    r1 =  cross_autocorrelation(polar_data, data_rolled)
    data_rolled = np.roll(polar_data,2*shift, axis=1)
    r2 = cross_autocorrelation(polar_data, data_rolled)
    return (r1+r2) / 2. 

def four_fold_symmetry(polar_data):
    """
    polar_data: image array in the polar coordinate system, (r, theta)
    """
    
    shape = polar_data.shape
    shift = int(shape[1] / 4)
    data_rolled = np.roll(polar_data,shift, axis=1)
    r1 = cross_autocorrelation(polar_data, data_rolled)
    data_rolled = np.roll(polar_data,2*shift, axis=1)
    r2 = cross_autocorrelation(polar_data, data_rolled)    
    data_rolled = np.roll(polar_data,3*shift, axis=1)
    r3 = cross_autocorrelation(polar_data, data_rolled)    
    return (r1+r2+r3) / 3.   
    
def six_fold_symmetry(polar_data):
    """
    polar_data: image array in the polar coordinate system, (r, theta)
    """
    
    shape = polar_data.shape
    shift = int(shape[1] / 6)
    data_rolled = np.roll(polar_data,shift, axis=1)
    r1 = cross_autocorrelation(polar_data, data_rolled)
    data_rolled = np.roll(polar_data,2*shift,axis=1)
    r2 = cross_autocorrelation(polar_data, data_rolled)
    data_rolled = np.roll(polar_data,3*shift,axis=1)
    r3 = cross_autocorrelation(polar_data, data_rolled)    
    data_rolled = np.roll(polar_data,4*shift,axis=1)
    r4 = cross_autocorrelation(polar_data, data_rolled)        
    data_rolled = np.roll(polar_data,5*shift,axis=1)
    r5 = cross_autocorrelation(polar_data, data_rolled)
    return (r1+r2+r3+r4+r5)/5.

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
        output *= r_i[:, np.newaxis]
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

