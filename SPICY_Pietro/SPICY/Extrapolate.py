# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:19:34 2022

@author: Pietro.Sperotto
"""
import numpy as np
from SPICY.Matrix import PHI,PHI3D
def Fit_vel2D(W_u,W_v,X_C,Y_C,XG,YG,c,rcond):
    """
        This method uses the weights to compute the scalar field in the 
        grid XG, YG.
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param W_u: array
                  weights of RBF for u
        :param W_v: array
                  weights of RBF for v
        :param X_C: array
                  One dimensional array of X position of collocation point
        :param Y_C: array
                  One dimensional array of Y position of collocation point
        :param XG: array
                  X position where the fit has to be estimated   
        :param YG: array
                  Y position where the fit has to be estimated
        :param c: array
                  shape parameter


        ----------------------------------------------------------------------------------------------------------------
        Returns
        -------

        :return: U array the fitted values in the point required
"""

    SHAPE=np.shape(XG)
    PHI_XX=PHI(X_C,Y_C,XG.reshape(-1),YG.reshape(-1),c,rcond)
    U1=PHI_XX.dot(W_u)
    V1=PHI_XX.dot(W_v)
    U=U1.reshape(SHAPE)
    V=V1.reshape(SHAPE)
    return U,V

def Fit_RBF(W,X_C,Y_C,XG,YG,c,rcond):
    """
        This method uses the weights to compute the scalar field in the 
        grid XG, YG.
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param W: array
                  weights of RBF
        :param X_C: array
                  One dimensional array of X position of collocation point
        :param Y_C: array
                  One dimensional array of Y position of collocation point
        :param XG: array
                  X position where the fit has to be estimated   
        :param YG: array
                  Y position where the fit has to be estimated
        :param c: array
                  shape parameter


        ----------------------------------------------------------------------------------------------------------------
        Returns
        -------

        :return: U array the fitted values in the point required
"""

    SHAPE=np.shape(XG)
    PHI_XX=PHI(X_C,Y_C,XG.reshape(-1),YG.reshape(-1),c,rcond)
    U1=PHI_XX.dot(W)
    U=U1.reshape(SHAPE)
    
    return U

def Fit_RBF3D(W,X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
        This method uses the weights to compute the scalar field in the 
        grid XG, YG,ZG.
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param W: array
                  weights of RBF
        :param X_C: array
                  One dimensional array of X position of collocation point
        :param Y_C: array
                  One dimensional array of Y position of collocation point
        :param Z_C: array
                  One dimensional array of Z position of collocation point
        :param XG: array
                  X position where the fit has to be estimated   
        :param YG: array
                  Y position where the fit has to be estimated
        :param ZG: array
                  Z position where the fit has to be estimated
        :param c: array
                  shape parameter


        ----------------------------------------------------------------------------------------------------------------
        Returns
        -------

        :return: U array the fitted values in the point required
"""

    SHAPE=np.shape(XG)
    PHI_XX=PHI3D(X_C,Y_C,Z_C,XG.reshape(-1),YG.reshape(-1),ZG.reshape(-1),c,rcond)
    U1=PHI_XX.dot(W)
    U=U1.reshape(SHAPE)
    
    return U


def Fit_vel3D(W_u,W_v,W_w,X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
        This method uses the weights to compute the velocity field in the 
        grid XG, YG, ZG.
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param W_u: array
                  weights of RBF for u
        :param W_v: array
                  weights of RBF for v
        :param W_w: array
                  weights of RBF for w
        :param X_C: array
                  One dimensional array of X position of collocation point
        :param Y_C: array
                  One dimensional array of Y position of collocation point
        :param Z_C: array
                  One dimensional array of Z position of collocation point
        :param XG: array
                  X position where the fit has to be estimated   
        :param YG: array
                  Y position where the fit has to be estimated
        :param ZG: array
                  Z position where the fit has to be estimated
        :param c: array
                  shape parameter


        ----------------------------------------------------------------------------------------------------------------
        Returns
        -------

        :return: U array the fitted values in the point required
"""

    SHAPE=np.shape(XG)
    PHI_XX=PHI3D(X_C,Y_C,Z_C,XG.reshape(-1),YG.reshape(-1),ZG.reshape(-1),c,rcond)
    U1=PHI_XX.dot(W_u)
    V1=PHI_XX.dot(W_v)
    W1=PHI_XX.dot(W_w)
    U=U1.reshape(SHAPE)
    V=V1.reshape(SHAPE)
    W=W1.reshape(SHAPE)
    
    return U,V,W