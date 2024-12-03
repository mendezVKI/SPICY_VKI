"""
Utilities functions to deal with the basis and things like clustering, clipping etc.

Authors: Manuel Ratz
"""

import numpy as np

# these functions are used for the clustering and collocation
from sklearn.neighbors import NearestNeighbors
# Function for the k means clustering
from sklearn.cluster import MiniBatchKMeans

from scipy.stats.qmc import Halton
from ..utils._extnumpy import meshgrid_ravel

def get_shape_parameter_and_diameter(r_mM, c_k, basis):
    """
    This function clips the shape parameters of the RBFs and assigns their diameters.

    Parameters
    ----------

    c_k : 1D numpy.ndarray
        Array containing the shape parameters of the RBFs
    r_mM : list
        Minimum and maximum radius of the RBFs
    basis : str
        Type of basis function, must be c4 or Gaussian

    Returns
    -------

    c_k : 1D numpy.ndarray
        Clipped shapes of the RBFs
    d_k : 1D numpy.ndarray
        Associated radius of the RBFs based on the clipped shapes

    """
    if basis == 'gauss':
        # Set the max and min values of c_k
        c_min = 1 / (r_mM[1]) * np.sqrt(np.log(2))
        c_max = 1 / (r_mM[0]) * np.sqrt(np.log(2))
        # crop to the minimum and maximum value
        c_k[c_k < c_min] = c_min
        c_k[c_k > c_max] = c_max
        # for plotting purposes, store also the diameters
        d_k = 2 / c_k * np.sqrt(np.log(2))

    elif basis == 'c4':
        c_min = r_mM[0] / np.sqrt(1 - 0.5 ** 0.2)
        c_max = r_mM[1] / np.sqrt(1 - 0.5 ** 0.2)
        # crop to the minimum and maximum value
        c_k[c_k < c_min] = c_min
        c_k[c_k > c_max] = c_max
        # for plotting purposes, store also the diameters
        d_k = 2 * c_k * np.sqrt(1 - 0.5 ** 0.2)

    else:
        # Leave other options for future implementations.
        print('This basis is currently not implemented')
        c_k = None
        d_k = None

    return c_k, d_k



def collocation_kmeans(points, n_K_l):
    """
    Function to perform the kmeans clustering based on spatial data

    Parameters
    ----------
    points : 2D numpy.ndarray
        Array containing the data which is used in the clustering

        * Shape (2, n_p) in 2D
        * Shape (3, n_p) in 3D

    amount : float
        Amount of points (average) within a cluster. If n_p = 20 and number = 2,
        then approximately 10 RBFs will be placed on this clustering level.

    Returns
    -------
    centers : 2D numpy.ndarray
        Array containing the cluster centers (collocation points). It has the
        same shape as points, except that it is n_b = n_p/number long in the second dimension

    sigma_level : 1d numpy.ndarray
        Array containing the cluster diameters. These are already modified to remove
        shapes of 0 and clusters that only contain 1 point.

    """

    # Here, we compute the number of clusters and perform KMeans minibatch clustering.
    # The cluster centers are our RBF collocation points, so we extract these after fitting
    clust = int(np.ceil(np.shape(points)[0] / n_K_l))
    model = MiniBatchKMeans(n_clusters=clust, random_state=0, n_init=10)
    y_P = model.fit_predict(points)
    collocation_points = model.cluster_centers_

    # Get the nearest neighbour of each center
    sigma_level = _get_shape(collocation_points)

    # Remove all clusters which either have a distance of
    # zero to the nearest neighbor (that would be the same RBF)
    # and the clusters with only one point in them
    count = np.bincount(y_P, minlength=clust)
    sigma_level[count == 1] = np.min(sigma_level)

    return collocation_points, sigma_level



def _get_shape(collocation_points):
    neighbors = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(collocation_points)
    distances, indices = neighbors.kneighbors(collocation_points)
    sigma_level = distances[:, 1]
    sigma_level[sigma_level == 0] = np.max(sigma_level[sigma_level != 0])

    return sigma_level


def collocation_regular(n_K_l, bounds, dimension):

    if dimension == '2D':
        x_min, x_max, y_min, y_max = bounds
        n_x_l = int(np.round(np.sqrt(n_K_l * (x_max - x_min) / (y_max - y_min))))
        # Compute the number of collocation points and create the points
        n_colloc_x = n_x_l
        n_colloc_y = int(np.round(n_x_l) * (y_max - y_min) / (x_max - x_min))
        x_C = np.linspace(x_min, x_max, n_colloc_x)
        y_C = np.linspace(y_min, y_max, n_colloc_y)
        X_C, Y_C = meshgrid_ravel(x_C, y_C)
        collocation_points = np.vstack((X_C, Y_C)).T

    elif dimension == '3D':
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        # Compute the number of collocation points and create the points
        n_colloc_x = n_x_l
        n_colloc_y = int(np.round(n_x_l) * (y_max - y_min) / (x_max - x_min))
        n_colloc_z = int(np.round(n_x_l) * (z_max - z_min) / (x_max - x_min))
        x_C = np.linspace(x_min, x_max, n_colloc_x)
        y_C = np.linspace(y_min, y_max, n_colloc_y)
        z_C = np.linspace(z_min, z_max, n_colloc_z)
        X_C, Y_C, Z_C = meshgrid_ravel(x_C, y_C, z_C)
        collocation_points = np.vstack((X_C, Y_C, Z_C)).T

    else:
        pass

    sigma_level = _get_shape(collocation_points)

    return collocation_points, sigma_level



def collocation_halton(n_K_l, bounds, dimension):

    if dimension == '2D':
        x_min, x_max, y_min, y_max = bounds

        sampler = Halton(d=2, scramble=True, seed=42)
        X_C, Y_C = sampler.random(n=n_K_l).T
        X_C = X_C * (x_max - x_min) + x_min
        Y_C = Y_C * (y_max - y_min) + y_min
        collocation_points = np.vstack((X_C, Y_C)).T

    elif dimension == '3D':
        x_min, x_max, y_min, y_max, z_min, z_max = bounds

        sampler = Halton(d=3, scramble=True, seed=42)
        X_C, Y_C, Z_C = sampler.random(n=n_K_l).T
        X_C = X_C * (x_max - x_min) + x_min
        Y_C = Y_C * (y_max - y_min) + y_min
        Z_C = Z_C * (z_max - z_min) + z_min
        collocation_points = np.vstack((X_C, Y_C, Z_C)).T

    else:
        pass

    sigma_level = _get_shape(collocation_points)

    return collocation_points, sigma_level

