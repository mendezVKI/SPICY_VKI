"""
Utilities functions to compute the harmonic basis in SPICY

Authors: Manuel Ratz
"""

import numpy as np

# =============================================================================
#  Harmonic functions in 2D
#  Includes: Phi_H_2D, Phi_H_2D_x, Phi_H_2D_y, Phi_H_2D_Laplacian
# =============================================================================

def Phi_H_2D(X_G, Y_G, n_hb):
    """
    Function to get the harmonic basis in 2D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H : 2D numpy.ndarray
        Basis matrix of shape (n_p, n_hb**4)

    """

    # The output is a matrix of size (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 4  # number of possible dispositions of the harmonic basis in R2.
    Phi_H = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y

    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        k_x_i = 2 * np.pi * (i_s[count] + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j_s[count] + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m_s[count] + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q_s[count] + 1)  # This goes with cosines

        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)

        # Assign the column of Phi_H
        Phi_H[:, count] = sin_k_i_x * cos_k_j_x * sin_k_m_y * cos_k_q_y
        count += 1

    return Phi_H


def Phi_H_2D_x(X_G, Y_G, n_hb):
    """
    Function to get the x-derivative of harmonic basis in 2D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H_x : 2D numpy.ndarray
        Derivative in x of basis matrix of shape (n_p, n_hb**4)

    """

    # Get the number of points
    n_p = len(X_G)

    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 4  # number of possible dispositions of the harmonic basis in R2.
    Phi_H_x = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y

    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_i_x = np.cos(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_j_x = np.sin(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        # Assign the column of Phi_H
        Prime = -(k_x_j * sin_k_i_x * sin_k_j_x - k_x_i * cos_k_i_x * cos_k_j_x)
        Phi_H_x[:, count] = Prime * sin_k_m_y * cos_k_q_y
        count += 1

    return Phi_H_x


def Phi_H_2D_y(X_G, Y_G, n_hb):
    """
    Function to get the y-derivative of harmonic basis in 2D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H_y : 2D numpy.ndarray
        Derivative in y of basis matrix of shape (n_p, n_hb**4)

    """

    # Get the number of points
    n_p = len(X_G)

    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 4  # number of possible dispositions of the harmonic basis in R2.
    Phi_H_y = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y

    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_m_y = np.cos(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_q_y = np.sin(k_y_q * Y_G)

        # Assign the column of Phi_H
        Prime = -(k_y_q * sin_k_m_y * sin_k_q_y - k_y_m * cos_k_m_y * cos_k_q_y)
        Phi_H_y[:, count] = Prime * sin_k_i_x * cos_k_j_x
        count += 1

    return Phi_H_y


def Phi_H_2D_Laplacian(X_G, Y_G, n_hb):
    """
    Function to get the Laplacian of harmonic basis in 2D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Lap_H : 2D numpy.ndarray
        Laplacian of basis matrix of shape (n_p, n_hb**4)

    """

    # number of points
    n_p = len(X_G)

    # The number of harmonic bases will be:
    n_h = n_hb ** 4  # number of possible dispositions of the harmonic basis in R2.
    Lap_H = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y

    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        k_x_i = 2 * np.pi * (i + 1) / 1  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1) / 1  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1) / 1  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1) / 1  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_i_x = np.cos(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_j_x = np.sin(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_m_y = np.cos(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_q_y = np.sin(k_y_q * Y_G)

        # Compute the derivatives of the harmonic basis sin_k_i_x
        phi_ijmq_xx = -sin_k_m_y * cos_k_q_y * (2 * k_x_i * k_x_j * cos_k_i_x * sin_k_j_x +
                                                (k_x_j ** 2 + k_x_i ** 2) * sin_k_i_x * cos_k_j_x)

        phi_ijmq_yy = -sin_k_i_x * cos_k_j_x * (2 * k_y_m * k_y_q * cos_k_m_y * sin_k_q_y +
                                                (k_y_q ** 2 + k_y_m ** 2) * sin_k_m_y * cos_k_q_y)
        # Assign the column of the Laplacian
        Lap_H[:, count] = phi_ijmq_xx + phi_ijmq_yy
        count += 1

    return Lap_H


def Phi_H_3D(X_G, Y_G, Z_G, n_hb):
    """
    Function to get the harmonic basis in 3D.

    Parameters
    ----------
    X_G, Y_G, Z_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H : 2D numpy.ndarray
        Derivative in x of basis matrix of shape (n_hb**6, n_p)

    """

    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 6  # number of possible dispositions of the harmonic basis in R2.
    Phi_H = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y*sin_k_r_z*sin_k_s_z

    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((5, 1, 2, 3, 4, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 5, 2, 3, 4, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 5, 3, 4, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 5, 4, 3)).ravel()
    r_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 5, 4)).ravel()
    s_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 4, 5)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        r = r_s[count]
        s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        k_y_r = 2 * np.pi * (r + 1)  # This goes with sines
        k_y_s = np.pi / 2 * (2 * s + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_r_z = np.sin(k_y_r * Z_G)
        cos_k_s_z = np.cos(k_y_s * Z_G)

        # Assign the column of Phi_H
        Phi_H[:, count] = sin_k_i_x * cos_k_j_x * sin_k_m_y * cos_k_q_y * sin_k_r_z * cos_k_s_z

    return Phi_H


def Phi_H_3D_x(X_G, Y_G, Z_G, n_hb):
    """
    Function to get the x-derivative of harmonic basis in 3D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H_x : 2D numpy.ndarray
        Derivative in x of basis matrix of shape (n_p, n_hb**6)

    """

    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 6  # number of possible dispositions of the harmonic basis in R2.
    Phi_H_x = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y*sin_k_r_z*sin_k_s_z

    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((5, 1, 2, 3, 4, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 5, 2, 3, 4, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 5, 3, 4, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 5, 4, 3)).ravel()
    r_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 5, 4)).ravel()
    s_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 4, 5)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        r = r_s[count]
        s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        k_z_r = 2 * np.pi * (r + 1)  # This goes with sines
        k_z_s = np.pi / 2 * (2 * s + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_i_x = np.cos(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_j_x = np.sin(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_r_z = np.sin(k_z_r * Z_G)
        cos_k_s_z = np.cos(k_z_s * Z_G)

        # Assign the column of Phi_H_x
        Prime = -(k_x_j * sin_k_i_x * sin_k_j_x - k_x_i * cos_k_i_x * cos_k_j_x)
        Phi_H_x[:, count] = Prime * sin_k_m_y * cos_k_q_y * sin_k_r_z * cos_k_s_z

    return Phi_H_x


def Phi_H_3D_y(X_G, Y_G, Z_G, n_hb):
    """
    Function to get the y-derivative of harmonic basis in 3D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H_y : 2D numpy.ndarray
        Derivative in y of basis matrix of shape (n_p, n_hb**6)

    """

    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 6  # number of possible dispositions of the harmonic basis in R2.
    Phi_H_y = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y*sin_k_r_z*sin_k_s_z

    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((5, 1, 2, 3, 4, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 5, 2, 3, 4, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 5, 3, 4, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 5, 4, 3)).ravel()
    r_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 5, 4)).ravel()
    s_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 4, 5)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        r = r_s[count]
        s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        k_z_r = 2 * np.pi * (r + 1)  # This goes with sines
        k_z_s = np.pi / 2 * (2 * s + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_m_y = np.cos(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_q_y = np.sin(k_y_q * Y_G)
        sin_k_r_z = np.sin(k_z_r * Z_G)
        cos_k_s_z = np.cos(k_z_s * Z_G)

        # Assign the column of Phi_H_y
        Prime = -(k_y_m * sin_k_m_y * sin_k_q_y - k_y_q * cos_k_m_y * cos_k_q_y)
        Phi_H_y[:, count] = Prime * sin_k_i_x * cos_k_j_x * sin_k_r_z * cos_k_s_z

    return Phi_H_y


def Phi_H_3D_z(X_G, Y_G, Z_G, n_hb):
    """
    Function to get the z-derivative of harmonic basis in 3D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Phi_H_z : 2D numpy.ndarray
        Derivative in z of basis matrix of shape (n_p, n_hb**6)

    """

    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 6  # number of possible dispositions of the harmonic basis in R2.
    Phi_H_z = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y*sin_k_r_z*sin_k_s_z

    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((5, 1, 2, 3, 4, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 5, 2, 3, 4, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 5, 3, 4, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 5, 4, 3)).ravel()
    r_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 5, 4)).ravel()
    s_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 4, 5)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        r = r_s[count]
        s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        k_z_r = 2 * np.pi * (r + 1)  # This goes with sines
        k_z_s = np.pi / 2 * (2 * s + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_r_z = np.sin(k_z_r * Z_G)
        cos_k_r_z = np.cos(k_z_r * Z_G)
        cos_k_s_z = np.cos(k_z_s * Z_G)
        sin_k_s_z = np.sin(k_z_s * Z_G)

        # Assign the column of Phi_H_z
        Prime = -(k_z_s * sin_k_r_z * sin_k_s_z - k_z_r * cos_k_r_z * cos_k_s_z)
        Phi_H_z[:, count] = Prime * sin_k_i_x * cos_k_j_x * sin_k_m_y * cos_k_q_y

    return Phi_H_z


def Phi_H_3D_Laplacian(X_G, Y_G, Z_G, n_hb):
    """
    Function to get the laplacian of harmonic basis in 3D.

    Parameters
    ----------
    X_G, Y_G : 1D numpy.ndarrays
        Evaluation coordinates of the basis.

    n_hb : int
        Number of harmonic bases to put.

    Returns
    -------
    Lap_H : 2D numpy.ndarray
        Laplacian of basis matrix of shape (n_p, n_hb**6)

    """

    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb ** 6  # number of possible dispositions of the harmonic basis in R2.
    Lap_H = np.zeros((n_p, n_h))
    # Developer note: the basis is:
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y*sin_k_r_z*sin_k_s_z

    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((5, 1, 2, 3, 4, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 5, 2, 3, 4, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 5, 3, 4, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 5, 4, 3)).ravel()
    r_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 5, 4)).ravel()
    s_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3, 4, 5)).ravel()

    for count in range(n_h):
        i = i_s[count]
        j = j_s[count]
        m = m_s[count]
        q = q_s[count]
        r = r_s[count]
        s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2 * np.pi * (i + 1)  # This goes with sines
        k_x_j = np.pi / 2 * (2 * j + 1)  # This goes with cosines
        k_y_m = 2 * np.pi * (m + 1)  # This goes with sines
        k_y_q = np.pi / 2 * (2 * q + 1)  # This goes with cosines
        k_z_r = 2 * np.pi * (r + 1)  # This goes with sines
        k_z_s = np.pi / 2 * (2 * s + 1)  # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i * X_G)
        cos_k_i_x = np.cos(k_x_i * X_G)
        cos_k_j_x = np.cos(k_x_j * X_G)
        sin_k_j_x = np.sin(k_x_j * X_G)
        sin_k_m_y = np.sin(k_y_m * Y_G)
        cos_k_m_y = np.cos(k_y_m * Y_G)
        cos_k_q_y = np.cos(k_y_q * Y_G)
        sin_k_q_y = np.sin(k_y_q * Y_G)
        sin_k_r_z = np.sin(k_z_r * Z_G)
        cos_k_r_z = np.cos(k_z_r * Z_G)
        cos_k_s_z = np.cos(k_z_s * Z_G)
        sin_k_s_z = np.sin(k_z_s * Z_G)

        # Compute the derivatives of the harmonic basis sin_k_i_x
        phi_ijmqrs_xx = (-sin_k_m_y * cos_k_q_y * sin_k_r_z * cos_k_s_z *
                         (2 * k_x_i * k_x_j * cos_k_i_x * sin_k_j_x +
                          (k_x_j ** 2 + k_x_i ** 2) * sin_k_i_x * cos_k_j_x))

        phi_ijmqrs_yy = (-sin_k_i_x * cos_k_j_x * sin_k_r_z * cos_k_s_z *
                         (2 * k_y_m * k_y_q * cos_k_m_y * sin_k_q_y +
                          (k_y_q ** 2 + k_y_m ** 2) * sin_k_m_y * cos_k_q_y))

        phi_ijmqrs_zz = (-sin_k_i_x * cos_k_j_x * sin_k_m_y * cos_k_q_y *
                         (2 * k_z_r * k_z_s * cos_k_r_z * sin_k_s_z +
                          (k_z_r ** 2 + k_z_s ** 2) * sin_k_r_z * cos_k_s_z))

        # Assign the column of the Laplacian
        Lap_H[:, count] = phi_ijmqrs_xx + phi_ijmqrs_yy + phi_ijmqrs_zz

    return Lap_H
