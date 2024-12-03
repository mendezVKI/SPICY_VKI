"""
Utilities functions to compute the RBF basis in SPICY

Authors: Manuel Ratz
"""

import numpy as np
from ..utils._extnumpy import square_distance

# =============================================================================
#  RBF functions in 2D
#  Includes: Phi_RBF_2D, Phi_RBF_2D_x, Phi_RBF_2D_y, Phi_RBF_2D_Laplacian
# =============================================================================

def Phi_RBF_2D(X_G, Y_G, X_C, Y_C, c_k, basis):
    """
    Function to get the radial basis in 2D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    X_C, Y_C : 1D numpy.ndarrays
        Collocation points of the basis elements.

    c_k : 1D numpy.ndarray
        Shape parameter of the basis elements

    basis : str
        Which RBF to use. Must be 'gauss' or 'c4'.

    Returns
    -------
    Phi_RBF : 2D numpy.ndarray
        Basis matrix of shape (n_p, n_b)

    """

    # This is the contribution of the RBF part
    n_b = len(X_C)
    n_p = len(X_G)
    Phi_RBF = np.zeros((n_p, n_b))

    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian = np.exp(-c_k[r]**2 * square_distance(X_G, X_C[r], Y_G, Y_C[r]))
            # Assemble into matrix
            Phi_RBF[:, r] = gaussian

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance and points inside
            d = np.sqrt(square_distance(X_G, X_C[r], Y_G, Y_C[r]))
            valid = np.abs(d) < c_k[r]
            # Compute Phi
            phi = (1+d[valid]/c_k[r])**5 * (1-d[valid]/c_k[r])**5
            # Assemble into matrix
            Phi_RBF[valid, r] = phi

    # Return the matrix
    return Phi_RBF


def Phi_RBF_2D_Deriv(X_G, Y_G, X_C, Y_C, c_k, basis, order=1):
    """
    Function to get the derivatives of the radial basis in 2D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    X_C, Y_C : 1D numpy.ndarrays
        Collocation points of the basis elements.

    c_k : 1D numpy.ndarray
        Shape parameter of the basis elements

    basis : str
        Which RBF to use. Must be 'gauss' or 'c4'.

    order : int, default 1
        The order of the derivative. order=1 corresponds to first derivative, etc.

    Returns
    -------
    RBF_Derivs : 2D numpy.ndarray
        Derivatives in all directions of basis matries, each of shape (n_p, n_b).
        The number of derivatives is varying depending on the order, since mixed
        derivatives are also computed

    """

    # number of bases (n_b) and points (n_p)
    n_b = len(X_C)
    n_p = len(X_G)
    RBF_Derivs = np.zeros((order+1, n_p, n_b))

    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            d_sq = square_distance(X_G, X_C[r], Y_G, Y_C[r])
            # Compute the Gaussian
            gaussian = 2*c_k[r]**2 * np.exp(-c_k[r] ** 2 * d_sq)
            # Assemble into matrix
            if order == 1:
                RBF_Derivs[0, :, r] = - gaussian * (X_G-X_C[r])
                RBF_Derivs[1, :, r] = - gaussian * (Y_G-Y_C[r])

            elif order == 2:
                RBF_Derivs[0, :, r] = gaussian * (2*(X_G - X_C[r])**2 * c_k[r]**2 - 1)
                RBF_Derivs[1, :, r] = gaussian * (2*(Y_G - Y_C[r])**2 * c_k[r]**2 - 1)
                RBF_Derivs[2, :, r] = gaussian * (X_G - X_C[r]) * (Y_G - Y_C[r]) * 2*c_k[r]**2

            else:
                raise ValueError('Derivatives higher than 2nd not implemented')

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            d = np.sqrt(square_distance(X_G, X_C[r], Y_G, Y_C[r]))
            valid = np.abs(d) < c_k[r]
            # Compute the factor
            if order == 1:
                factor = -10/c_k[r]**10 * (c_k[r]+d[valid])**4 * (c_k[r]-d[valid])**4
                RBF_Derivs[0, valid, r] = factor * (X_G[valid]-X_C[r])
                RBF_Derivs[1, valid, r] = factor * (Y_G[valid]-Y_C[r])

            elif order == 2:
                factor = -10/c_k[r]**10 * (c_k[r]+d[valid])**3 * (c_k[r]-d[valid])**3
                RBF_Derivs[0, valid, r] = factor * (-8*(X_G[valid]-X_C[r])**2+c_k[r]**2-d[valid]**2)
                RBF_Derivs[1, valid, r] = factor * (-8*(Y_G[valid]-Y_C[r])**2+c_k[r]**2-d[valid]**2)
                RBF_Derivs[2, valid, r] = -8*factor * (Y_G[valid]-Y_C[r]) * (X_G[valid]-X_C[r])

            else:
                raise ValueError('Derivatives higher than 2nd not implemented')

    else:
        raise ValueError('Not implemented error')

    if order == 1:
        return RBF_Derivs[0, :, :], RBF_Derivs[1, :, :]
    if order == 2:
        return RBF_Derivs[0, :, :], RBF_Derivs[1, :, :], RBF_Derivs[2, :, :]

    return


# =============================================================================
#  RBF functions in 3D
#  Includes: Phi_RBF_3D, Phi_RBF_3D_Deriv
# =============================================================================

def Phi_RBF_3D(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
    """
    Function to get the radial basis in 2D.

    Parameters
    ----------
    X_G, Y_G, Z_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    X_C, Y_C, Z_C : 1D numpy.ndarrays
        Collocation points of the basis elements.

    c_k : 1D numpy.ndarray
        Shape parameter of the basis elements

    basis : str
        Which RBF to use. Must be 'gauss' or 'c4'.

    Returns
    -------
    Phi_RBF : 2D numpy.ndarray
        Basis matrix of shape (n_p, n_b)

    """

    # This is the contribution of the RBF part
    n_b = len(X_C)
    n_p = len(X_G)
    Phi_RBF = np.zeros((n_p, n_b))

    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the distance
            d_sq = square_distance(X_G, X_C[r], Y_G, Y_C[r], Z_G, Z_C[r])
            # Assemble into matrix
            Phi_RBF[:, r] = np.exp(-c_k[r]**2*d_sq)

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance and points inside
            d = np.sqrt(square_distance(X_G, X_C[r], Y_G, Y_C[r], Z_G, Z_C[r]))
            valid = np.abs(d) < c_k[r]
            # Compute Phi
            phi = (1+d[valid]/c_k[r])**5 * (1-d[valid]/c_k[r])**5
            # Assemble into matrix
            Phi_RBF[valid, r] = phi

    return Phi_RBF


def Phi_RBF_3D_Deriv(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis, order=1):
    """
    Function to get the derivatives of the radial basis in 3D.

    Parameters
    ----------
    X_G, Y_G, Z_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    X_C, Y_C, Z_C : 1D numpy.ndarrays
        Collocation points of the basis elements.

    c_k : 1D numpy.ndarray
        Shape parameter of the basis elements

    basis : str
        Which RBF to use. Must be 'gauss' or 'c4'.

    order : int, default 1
        The order of the derivative. order=1 corresponds to first derivative, etc.

    Returns
    -------
    RBF_Derivs : 2D numpy.ndarray
        Derivatives in all directions of basis matries, each of shape (n_p, n_b).
        The number of derivatives is varying depending on the order, since mixed
        derivatives are also computed

    """

    # number of bases (n_b) and points (n_p)
    n_b = len(X_C)
    n_p = len(X_G)

    RBF_Derivs = np.zeros((3*order, n_p, n_b))

    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            d_sq = square_distance(X_G, X_C[r], Y_G, Y_C[r], Z_G, Z_C[r])
            # Compute the Gaussian
            gaussian = 2*c_k[r]**2 * np.exp(-c_k[r] ** 2 * d_sq)
            # Assemble into matrix
            if order == 1:
                RBF_Derivs[0, :, r] = - gaussian * (X_G-X_C[r])
                RBF_Derivs[1, :, r] = - gaussian * (Y_G-Y_C[r])
                RBF_Derivs[2, :, r] = - gaussian * (Z_G-Z_C[r])

            elif order == 2:
                RBF_Derivs[0, :, r] = gaussian * (2*(X_G - X_C[r])**2 * c_k[r]**2 - 1)
                RBF_Derivs[1, :, r] = gaussian * (2*(Y_G - Y_C[r])**2 * c_k[r]**2 - 1)
                RBF_Derivs[2, :, r] = gaussian * (2*(Z_G - Z_C[r])**2 * c_k[r]**2 - 1)
                RBF_Derivs[3, :, r] = gaussian * (X_G - X_C[r]) * (Y_G - Y_C[r]) * 2*c_k[r]**2
                RBF_Derivs[4, :, r] = gaussian * (X_G - X_C[r]) * (Z_G - Z_C[r]) * 2*c_k[r]**2
                RBF_Derivs[5, :, r] = gaussian * (Y_G - Y_C[r]) * (Z_G - Z_C[r]) * 2*c_k[r]**2

            else:
                raise ValueError('Derivatives higher than 2nd not implemented')

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            d = np.sqrt(square_distance(X_G, X_C[r], Y_G, Y_C[r], Z_G, Z_C[r]))
            valid = np.abs(d) < c_k[r]
            # Compute the factor
            if order == 1:
                factor = -10/c_k[r]**10 * (c_k[r]+d[valid])**4 * (c_k[r]-d[valid])**4
                RBF_Derivs[0, valid, r] = factor * (X_G[valid]-X_C[r])
                RBF_Derivs[1, valid, r] = factor * (Y_G[valid]-Y_C[r])
                RBF_Derivs[2, valid, r] = factor * (Z_G[valid]-Z_C[r])

            elif order == 2:
                factor = -10/c_k[r]**10 * (c_k[r]+d[valid])**3 * (c_k[r]-d[valid])**3
                RBF_Derivs[0, valid, r] = factor * (-8*(X_G[valid]-X_C[r])**2 + c_k[r]**2-d[valid]**2)
                RBF_Derivs[1, valid, r] = factor * (-8*(Y_G[valid]-Y_C[r])**2 + c_k[r]**2-d[valid]**2)
                RBF_Derivs[2, valid, r] = factor * (-8*(Z_G[valid]-Z_C[r])**2 + c_k[r]**2-d[valid]**2)
                RBF_Derivs[3, valid, r] = -8*factor * (X_G[valid]-X_C[r]) * (Y_G[valid]-Y_C[r])
                RBF_Derivs[4, valid, r] = -8*factor * (X_G[valid]-X_C[r]) * (Z_G[valid]-Z_C[r])
                RBF_Derivs[5, valid, r] = -8*factor * (Y_G[valid]-Y_C[r]) * (Z_G[valid]-Z_C[r])

            else:
                raise ValueError('Derivatives higher than 2nd not implemented')

    else:
        raise ValueError('Not implemented error')

    if order == 1:
        return RBF_Derivs[0, :, :], RBF_Derivs[1, :, :], RBF_Derivs[2, :, :]
    if order == 2:
        return RBF_Derivs[0, :, :], RBF_Derivs[1, :, :], RBF_Derivs[2, :, :],\
            RBF_Derivs[3, :, :], RBF_Derivs[4, :, :], RBF_Derivs[5, :, :]
