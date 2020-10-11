import numpy as np


def calculate_Fisher_separability(features,classes):
    
    """
    Calculating Fishser's separability for every feature
    The definition of Fisher's separability
    
    separability = \frac{\sum_{i=1^n N_i*(\mu_i - \mu)^2 }}{\sum_{i=1}^n N_i * \sigma^2_i}
    i:         class label going from 1 to n. 
    N_i:       the number of samples in class i.
    \mu_i:     the sample mean of class i. 
    \mu        the overall sample mean
    sigma^2_i: sample variance of class i. 
               set 'Delta Degrees of Freedom' in numpy.var to zero (the default value) 
               in order to get the same value with the definition of Fisher separability in other places, 
               e.g., https://multivariatestatsjl.readthedocs.io/en/latest/mclda.html. 


    Input
    ----------------
    features.........2D array, (n_samples, n_features)
    classes..........1D array, (n_samples,)
    

    Output
    ----------------------
    separability........1D array(n_features,)

    """
    features = np.array(features)
    if not features.ndim == 2:
        raise ValueError('features should be 2D array')
    if not len(features) == len(classes):
        raise ValueError('length of features and calsses should be identical')
    labels = np.unique(classes)
    N_i  = []
    mu_i = []
    sigma2_i = []
    for label in labels:
        N_i.append(np.sum(classes==label))
        indices = np.where(classes==label)
        feature_i = features[indices]
        mu_i.append(np.mean(feature_i,axis=0))
        sigma2_i.append(np.var(feature_i, axis=0))
    N_i = np.expand_dims(np.array(N_i),axis=1)
    mu_i = np.array(mu_i)
    sigma2_i = np.array(sigma2_i)
    mu = np.expand_dims(np.mean(features,axis=0), axis=0)
    
    between = np.sum(N_i*(mu_i-mu)**2,axis=0)
    within = np.sum(N_i*sigma2_i, axis=0)
    return between/within 
