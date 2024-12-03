"""
Utilities functions to do various array stacking/padding operations that are not by default in numpy

Authors: Manuel Ratz
"""

import numpy as np


def square_distance(x1, x2, y1, y2, z1=None, z2=None):
    """
    Function to compute the distance between a set of 2D or 3D points.
    For a 2D problem, 'z1' and 'z2' should be given as None.

    Parameters
    ----------

    x1, y1, z1 : 1D np.ndarrays
        Coordinates of the first vector.

    x2, y2, z2 : 1D np.ndarrays
        Coordinates of the second vector.


    Returns
    -------

    d_sq : 1D np.ndarray
        Square distance between the two vectors.

    """
    if (z1 is not None and z2 is None) or (z1 is None and z2 is not None):
        raise ValueError('Inputs \'z1\' and \'z2\' must either both be 1d arrays or both be None')

    if z1 is None and z2 is None:
        d_sq = (x1-x2)**2 + (y1-y2)**2

    else:
        d_sq = (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2

    return d_sq


def stack_to_block_matrix(dimension, matrix):
    """
    Function to create a diagonal block matrix.

    Parameters
    ----------
    dimension : str
        Dimension of the problem, currently either '2D' or '3D'.

    matrix : 2D numpy.ndarray
        Matrix block that is stacked into the structure.


    Returns
    -------

    return_matrix : 2D numpy.ndarray
        Diagonal block matrix with either 4 or 9 blocks in case of 2D or 3D.

    """

    # Get the shape for stacking
    shape = matrix.shape

    # Depending on the dimension create the matrix using np.block
    if dimension == '2D':
        return_matrix = np.block([
            [matrix, np.zeros(shape)],
            [np.zeros(shape), matrix]
        ])

    elif dimension == '3D':
        return_matrix = np.block([
            [matrix, np.zeros(shape), np.zeros(shape)],
            [np.zeros(shape), matrix, np.zeros(shape)],
            [np.zeros(shape), np.zeros(shape), matrix]
        ])
    else:
        return_matrix = None

    return return_matrix


def meshgrid_ravel(x, y, z=None):
    """
    Utilities function that returns raveled numpy arrays from meshgrid.
    For a 2D problem, 'z' should be given as None.

    Parameters
    ----------

    x, y, z : 1D numpy.ndarray
        Arrays from numpy.linspace containing the points to be stacked


    Returns
    -------

    X, Y, Z : 1D numpy.ndarray
        Arrays created through linspace, flattened in row-major (C-style) formatting.

    """

    # This is a 2D case
    if z is None:
        X, Y = np.meshgrid(x, y)

        return X.ravel(), Y.ravel()

    # And this 3D
    else:
        Y, Z, X = np.meshgrid(y, z, x)

        return X.ravel(), Y.ravel(), Z.ravel()


def get_grid_spacing(arr):
    """
    Utilities function to get the grid spacing of a 1D numpy array.

    Parameters
    ----------

    arr : 1D numpy.ndarray
        Data array, typically created with np.linspace


    Returns
    -------

    spacing : float
        Distance between the array elements.

    """

    if arr.shape[0] == 1:
        spacing = 0
    else:
        spacing = arr[1] - arr[0]

    return spacing