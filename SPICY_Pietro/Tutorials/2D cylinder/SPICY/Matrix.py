# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 10:30:08 2021

@author: pietr
"""
import numpy as np
from scipy.special import erf
#PHI matrix of gaussians
def PHI(X_C,Y_C,XG,YG,c,rcond):
    """
   Create phi's matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Phi_x: array, matrix PHI
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Phi_x=np.zeros((n_p,nn+10))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Phi_x[:,0]=1
    Phi_x[:,1]=XG
    Phi_x[:,2]=YG
    Phi_x[:,3]=XG*YG
    Phi_x[:,4]=XG**2
    Phi_x[:,5]=YG**2
    Phi_x[:,6]=XG*YG**2
    Phi_x[:,7]=(XG**2)*YG
    Phi_x[:,8]=YG**3
    Phi_x[:,9]=XG**3
    #filling RBF cell    
    for i in range(10,nn+10):
        Phi_x[:,i]=np.exp(-c[i-10]**2*((X_C[i-10]-XG)**2+(Y_C[i-10]-YG)**2))
    return Phi_x
#3D Gaussian Matrix
def PHI3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create phi's matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Phi_x: array, matrix PHI
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Phi_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Phi_x[:,0]=1
    Phi_x[:,1]=XG
    Phi_x[:,2]=YG
    Phi_x[:,3]=ZG
    Phi_x[:,4]=XG*YG
    Phi_x[:,5]=XG*ZG
    Phi_x[:,6]=YG*ZG
    Phi_x[:,7]=XG**2
    Phi_x[:,8]=YG**2
    Phi_x[:,9]=ZG**2
    Phi_x[:,10]=XG**2*ZG
    Phi_x[:,11]=XG**2*YG
    Phi_x[:,12]=YG**2*XG
    Phi_x[:,13]=YG**2*ZG
    Phi_x[:,14]=ZG**2*XG
    Phi_x[:,15]=ZG**2*YG
    Phi_x[:,16]=XG**3
    Phi_x[:,17]=YG**3
    Phi_x[:,18]=ZG**3
    Phi_x[:,19]=ZG*XG*YG
    #filling RBF cell    
    for i in range(20,nn+20):
        Phi_x[:,i]=np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Phi_x
def Der_RBF_X3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DX matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DX
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Der_x[:,1]=1
    Der_x[:,4]=YG
    Der_x[:,5]=ZG
    Der_x[:,7]=2*XG
    Der_x[:,10]=2*XG*ZG
    Der_x[:,11]=2*XG*YG
    Der_x[:,12]=YG**2
    Der_x[:,14]=ZG**2
    Der_x[:,16]=3*XG**2
    Der_x[:,19]=YG*ZG
    #filling RBF cell    
    for i in range(20,nn+20):
        Der_x[:,i]=-2*c[i-20]**2*(-X_C[i-20]+XG)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Der_x
def Der_RBF_Y3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DY matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DY
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Der_x[:,2]=1
    Der_x[:,4]=XG
    Der_x[:,6]=ZG
    Der_x[:,8]=2*YG
    Der_x[:,11]=XG**2
    Der_x[:,12]=2*YG*XG
    Der_x[:,13]=2*YG*ZG
    Der_x[:,15]=ZG**2
    Der_x[:,17]=3*YG**2
    Der_x[:,19]=XG*ZG
    #filling RBF cell    
    for i in range(20,nn+20):
        Der_x[:,i]=-2*c[i-20]**2*(-Y_C[i-20]+YG)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Der_x
def Der_RBF_Z3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DZ matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DZ
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Der_x[:,3]=1
    Der_x[:,5]=XG
    Der_x[:,6]=YG
    Der_x[:,9]=2*ZG
    Der_x[:,10]=XG**2
    Der_x[:,13]=YG**2
    Der_x[:,14]=2*ZG*XG
    Der_x[:,15]=2*ZG*YG
    Der_x[:,18]=3*ZG**2
    Der_x[:,19]=XG*YG
    #filling RBF cell    
    for i in range(20,nn+20):
        Der_x[:,i]=-2*c[i-20]**2*(-Z_C[i-20]+ZG)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Der_x
#first derivative in respect of X of the basis
def Der_RBF_X(X_C,Y_C,XG,YG,c,rcond):
    """
   Create derivatives matrix in x of Phi
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DX differentiation
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+10))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Der_x[:,1]=1
    Der_x[:,3]=YG
    Der_x[:,4]=2*XG
    Der_x[:,6]=YG**2
    Der_x[:,7]=2*YG*XG
    Der_x[:,9]=3*XG**2
    #filling RBF cell
    for i in range(10,nn+10):
        Der_x[:,i]=-2*c[i-10]**2*(-X_C[i-10]+XG)*np.exp(-c[i-10]**2*((X_C[i-10]-XG)**2+(Y_C[i-10]-YG)**2))
    

    return Der_x

#first derivative in respect of Y of the basis
def Der_RBF_Y(X_C,Y_C,XG,YG,c,rcond):
    """
   Create derivatives matrix in y of Phi
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_y: array, matrix DY differentiation
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_y=np.zeros((n_p,nn+10))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Der_y[:,2]=1
    Der_y[:,3]=XG
    Der_y[:,5]=2*YG
    Der_y[:,6]=2*YG*XG
    Der_y[:,7]=XG**2
    Der_y[:,8]=3*YG**2
    #filling RBF cell
    for i in range(10,nn+10):
        Der_y[:,i]=-2*c[i-10]**2*(-Y_C[i-10]+YG)*np.exp(-c[i-10]**2*((X_C[i-10]-XG)**2+(Y_C[i-10]-YG)**2))
    return Der_y
def LAP_RBF(X_C,Y_C,XG,YG,c,rcond):
    """
   Create laplacian matrix of phi
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Lap_x: array, matrix laplacian with polynomials term
   """ 
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Lap_x=np.zeros((n_p,nn+10))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Lap_x[:,4]=2
    Lap_x[:,5]=2
    Lap_x[:,6]=2*XG
    Lap_x[:,7]=2*YG
    Lap_x[:,8]=6*YG
    Lap_x[:,9]=6*XG
    #filling RBF cell
    for i in range(10,nn+10):
        Lap_x[:,i]=((4*c[i-10]**4*((X_C[i-10]-XG)**2)-2*c[i-10]**2)*np.exp(-c[i-10]**2*((X_C[i-10]-XG)**2+(Y_C[i-10]-YG)**2))+(4*c[i-10]**4*((Y_C[i-10]-YG)**2)-2*c[i-10]**2)*np.exp(-c[i-10]**2*((X_C[i-10]-XG)**2+(Y_C[i-10]-YG)**2)))
    return Lap_x
def Der_RBF_XZ3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DXZ matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DXZ
    """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Der_x[:,5]=1
    Der_x[:,10]=2*XG
    Der_x[:,14]=2*ZG
    Der_x[:,19]=YG
    #filling RBF cell    
    for i in range(20,nn+20):
        Der_x[:,i]=(-2*c[i-20]**2*(-Z_C[i-20]+ZG))*(-2*c[i-20]**2*(-X_C[i-20]+XG))*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    
    return Der_x
def Der_RBF_XY3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DXY matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DY
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Der_x[:,4]=1
    Der_x[:,11]=2*XG
    Der_x[:,12]=2*YG
    Der_x[:,19]=ZG
    #filling RBF cell    
    for i in range(20,nn+20):
        Der_x[:,i]=(-2*c[i-20]**2*(-Y_C[i-20]+YG))*(-2*c[i-20]**2*(-X_C[i-20]+XG))*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    
    return Der_x
def Der_RBF_YZ3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DYZ matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Der_x: array, matrix DZ
   """  
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Der_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+10 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    Der_x[:,6]=1
    Der_x[:,13]=2*YG
    Der_x[:,15]=2*ZG
    Der_x[:,19]=XG
    #filling RBF cell    
    for i in range(20,nn+20):
        Der_x[:,i]=(-2*c[i-20]**2*(-Z_C[i-20]+ZG))*(-2*c[i-20]**2*(-Y_C[i-20]+YG))*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Der_x
def LAP_RBF3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create laplacian matrix of phi
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Lap_x: array, matrix laplacian with polynomials term
   """ 
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Lap_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Lap_x[:,7]=2
    Lap_x[:,8]=2
    Lap_x[:,9]=2
    Lap_x[:,10]=2*ZG
    Lap_x[:,11]=2*YG
    Lap_x[:,12]=2*XG
    Lap_x[:,13]=2*ZG
    Lap_x[:,14]=2*XG
    Lap_x[:,15]=2*YG
    Lap_x[:,16]=6*XG
    Lap_x[:,17]=6*YG
    Lap_x[:,18]=6*ZG
    #filling RBF cell
    for i in range(20,nn+20):
        Lap_x[:,i]=((4*c[i-20]**4*((X_C[i-20]-XG)**2)-2*c[i-20]**2)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))+(4*c[i-20]**4*((Y_C[i-20]-YG)**2)-2*c[i-20]**2)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2)))+(4*c[i-20]**4*((Z_C[i-20]-ZG)**2)-2*c[i-20]**2)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Lap_x
def Der_RBFXX3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DXX
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Lap_x: array, matrix DXX
   """ 
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Lap_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Lap_x[:,7]=2
    Lap_x[:,10]=2*ZG
    Lap_x[:,11]=2*YG
    Lap_x[:,16]=6*XG
    #filling RBF cell
    for i in range(20,nn+20):
        Lap_x[:,i]=(4*c[i-20]**4*((X_C[i-20]-XG)**2)-2*c[i-20]**2)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Lap_x
def Der_RBFYY3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DYY
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Lap_x: array, matrix DYY
   """ 
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Lap_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Lap_x[:,8]=2
    Lap_x[:,12]=2*XG
    Lap_x[:,13]=2*ZG
    Lap_x[:,17]=6*YG
    #filling RBF cell
    for i in range(20,nn+20):
        Lap_x[:,i]=(4*c[i-20]**4*((Y_C[i-20]-YG)**2)-2*c[i-20]**2)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Lap_x
def Der_RBFZZ3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create DZZ matrix
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Lap_x: array, DZZ
   """ 
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Lap_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Lap_x[:,9]=2
    Lap_x[:,14]=2*XG
    Lap_x[:,15]=2*YG
    Lap_x[:,18]=6*ZG
    #filling RBF cell
    for i in range(20,nn+20):
        Lap_x[:,i]=(4*c[i-20]**4*((Z_C[i-20]-ZG)**2)-2*c[i-20]**2)*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2))
    return Lap_x
def LAP_RBF3D_intZ(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond):
    """
   Create integral in Z od the laplacian matrix of phi
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param X_C: array, one dimensional array of collocation point X
     
     :param Y_C: array, one dimensional array of collocation point Y
     
     :param Z_C: array, one dimensional array of collocation point Z
                        
     :param XG: array, one dimensional array of all the grid point in X 
                        
     :param YG: array, one dimensional array of all the grid point in Y 
     
     :param ZG: array, one dimensional array of all the grid point in Z 
 
     :param c: array, one dimensional array of all the shape factor
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     
     :param rcond: float (optional), inverse of the maximum conditioning acceptable
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return Lap_x: array, integral in Z of the laplacian
   """ 
    #X_C,Y_C are control points
    #XG,YG are grid points
    #c is the shape factor
    n_p=len(XG)#number of row of the final matrix
    nn=len(X_C)#number of RBF which is used to interpolate
    Lap_x=np.zeros((n_p,nn+20))#allocating the memory for the matrix nn+4 because to the RBF a third order polynomial is used
    #filling the polynomials cell
    #the first two polinomial base are not filled because they are zero
    Lap_x[:,7]=2*XG
    Lap_x[:,8]=2*XG
    Lap_x[:,9]=2*XG
    Lap_x[:,10]=2*XG*ZG
    Lap_x[:,11]=2*YG*XG
    Lap_x[:,12]=XG**2
    Lap_x[:,13]=2*XG*ZG
    Lap_x[:,14]=XG**2
    Lap_x[:,15]=2*YG*XG
    Lap_x[:,16]=3*XG**2
    Lap_x[:,17]=6*YG*XG
    Lap_x[:,18]=6*ZG*XG
    #filling RBF cell
    for i in range(20,nn+20):
        Lap_x[:,i]= (-2*c[i-20]**2*(-X_C[i-20]+XG))*np.exp(-c[i-20]**2*((X_C[i-20]-XG)**2+(Y_C[i-20]-YG)**2+(Z_C[i-20]-ZG)**2)) + 2*c[i-20]*np.exp(-c[i-20]**2*((Z_C[i-20]-ZG)**2+(Y_C[i-20]-YG)**2))*(-1+c[i-20]**2*((ZG-Z_C[i-20])**2+(YG-Y_C[i-20])**2))*np.sqrt(np.pi)*erf(c[i-20]*(XG-X_C[i-20]))
    return Lap_x
##From here there are thing that has been deprecated
def OPT_MAT(PHI,PHI_CON,NRBF,NCON,CON,f,fmax,alpha):
    """
    Create the matrix and known term of the constrained interpolation problem
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param PHI: array, matrix PHI or the laplacian of PHI
     
     :param PHI_CON: array, matrix PHI evaluated in constraint points
                        
     :param NRBF: int, number of RBF used
                        
     :param NCON: int, number of constraint used
 
     :param CON: array,constraint values
     
     :param f: array,values  to interpolate
     
     :param fmax: float,maximum value of f
     
     :param alpha: float,maximum value of alpha
     ----------------------------------------------------------------------------------------------------------------
     Returns 
     -------

     :return A: array, matrix to solve
     :return b: array, known term of the matrix
   """ 
    A1=np.hstack((2*PHI.T.dot(PHI)+2*alpha*np.eye(NRBF+10),PHI_CON.T))#the plus 10 is cuz of the polynomials which are added to the RBF requested 
    A2=np.hstack((PHI_CON,np.zeros((NCON,NCON))))
    A=np.vstack((A1,A2))
    b=np.hstack((2*PHI.T.dot(f/fmax),CON/fmax))
    return A,b
def OPT_MAT3D(PHI,PHI_CON,NRBF,NCON,CON,f,fmax,alpha):
    """
    Create the matrix and known term of the constrained interpolation problem
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param PHI: array, matrix PHI or the laplacian of PHI
     
     :param PHI_CON: array, matrix PHI evaluated in constraint points
                        
     :param NRBF: int, number of RBF used
                        
     :param NCON: int, number of constraint used
 
     :param CON: array,constraint values
     
     :param f: array,values  to interpolate
     
     :param fmax: float,maximum value of f
     
     :param alpha: float,maximum value of alpha
     ----------------------------------------------------------------------------------------------------------------
     Returns 
     -------

     :return A: array, matrix to solve
     :return b: array, known term of the matrix
   """ 
   
    A=2*PHI.T.dot(PHI)+2*alpha*np.eye(NRBF+20)#the plus 10 is cuz of the polynomials which are added to the RBF requested 
    B=PHI_CON.T
    b1=2*PHI.T.dot(f/fmax)
    b2=CON/fmax
    return A,B,b1,b2

def OPT_MAT_DIV_FREE(PHI,DX,DY,PHI_CON,DX_CON,DY_CON,NRBF,NCON,NDIV,CONu,CONv,u,v,umax,vmax,alpha,DIV):
    """
    Create the matrix and known term of the constrained interpolation problem with divergence free
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param PHI: array, matrix PHI or the laplacian of PHI
     
     :param DX: array,DX derivative matrix in the no divergence places
     
     :param DY: array,DY derivative matrix in the no divergence places
     
     :param PHI_CON: array, matrix PHI evaluated in constraint points
     
     :param DX_CON: array,DX derivative matrix in the no divergence places
     
     :param DY_CON: array,DY derivative matrix in the no divergence places
                        
     :param NRBF: int, number of RBF used
                        
     :param NCON: int, number of constraint used
     
     :param NDIV: int, number of point in which divergence free is applied
 
     :param CONu: array,constraint values on u
     
     :param CONv: array,constraint values on v
     
     :param u: array,values of u
     
     :param u: array,values of v
     
     :param umax: float,maximum value of u
     
     :param vmax: float,maximum value of v
     
     :param alpha: float,maximum value of alpha
     ----------------------------------------------------------------------------------------------------------------
     Returns 
     -------

     :return A: array, matrix to solve
     :return b:known term
    """
    MAX=max((umax,vmax))
    A1=np.hstack(((2*PHI.T.dot(PHI)+2*alpha*np.eye(NRBF+10))+DIV*2*DX.T.dot(DX),DIV*2*DX.T.dot(DY),PHI_CON.T,np.zeros((NRBF+10,NCON)),DX_CON.T))
    A2=np.hstack((DIV*2*DY.T.dot(DX),(2*PHI.T.dot(PHI)+2*alpha*np.eye(NRBF+10))+DIV*2*DY.T.dot(DY),np.zeros((NRBF+10,NCON)),PHI_CON.T,DY_CON.T))           
    A3=np.hstack((PHI_CON,np.zeros((NCON,2*NCON+NRBF+10+NDIV))))
    A4=np.hstack((np.zeros((NCON,NRBF+10)),PHI_CON,np.zeros((NCON,2*NCON+NDIV))))
    A5=np.hstack((DX_CON,DY_CON,np.zeros((NDIV,2*NCON+NDIV))))

    A=np.vstack((A1,A2,A3,A4,A5))
    b=np.hstack((2*PHI.T.dot(u),2*PHI.T.dot(v),CONu,CONv,np.zeros(NDIV)))
    return A,b/MAX
def OPT_MAT_DIV_FREE3D(PHI,DX,DY,DZ,PHI_CON,DX_CON,DY_CON,DZ_CON,NRBF,NCON,NDIV,CONu,CONv,CONw,u,v,w,umax,vmax,wmax,alpha,DIV):
    """
    Create the matrix and known term of the constrained interpolation problem with divergence free
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param PHI: array, matrix PHI or the laplacian of PHI     
     
     :param DX: array,DX derivative matrix in the no divergence places
     
     :param DY: array,DY derivative matrix in the no divergence places
     
     :param DZ: array,DZ derivative matrix in the no divergence places
     
     :param PHI_CON: array, matrix PHI evaluated in constraint points
     
     :param DX_CON: array,DX derivative matrix in the no divergence places
     
     :param DY_CON: array,DY derivative matrix in the no divergence places
     
     :param DZ_CON: array,DZ derivative matrix in the no divergence places
                        
     :param NRBF: int, number of RBF used
                        
     :param NCON: int, number of constraint used
     
     :param NDIV: int, number of point in which divergence free is applied
 
     :param CONu: array,constraint values on u
     
     :param CONv: array,constraint values on v
     
     :param CONw: array,constraint values on w
     
     :param u: array,values of u
     
     :param v: array,values of v
     
     :param w: array,values of w
     
     :param umax: float,maximum value of u
     
     :param vmax: float,maximum value of v
     
     :param wmax: float,maximum value of w
     
     :param alpha: float,maximum value of alpha
     ----------------------------------------------------------------------------------------------------------------
     Returns 
     -------

     :return A: array, matrix to solve A
     
     :return B: array, matrix to solve B
     
     :return b1:known term
     
     :return b2:known term
    """
    MAX=max((umax,vmax,wmax))
    b1=np.hstack((2*PHI.T.dot(u/(MAX)),2*PHI.T.dot(v/(MAX)),2*PHI.T.dot(w/(MAX))))
    PHITPHI=2*PHI.T.dot(PHI)
    del PHI
    DXTDY=2*DX.T.dot(DY)
    DXTDZ=2*DX.T.dot(DZ)
    DYTDZ=2*DY.T.dot(DZ)
    A1=np.hstack(((PHITPHI+2*alpha*np.eye(NRBF+20))+DIV*2*DX.T.dot(DX),DIV*DXTDY,DIV*DXTDZ))
    A2=np.hstack((DIV*DXTDY.T,(PHITPHI+2*alpha*np.eye(NRBF+20))+DIV*2*DY.T.dot(DY),DIV*DYTDZ))
    A3=np.hstack((DIV*DXTDZ.T,+DIV*DYTDZ.T,DIV*2*DZ.T.dot(DZ)+(PHITPHI+2*alpha*np.eye(NRBF+20)))) 
    A=np.vstack((A1,A2,A3))
    del DX,DZ,DY,DXTDY,DXTDZ,DYTDZ,PHITPHI,A1,A2,A3
    B1=np.hstack((PHI_CON.T,np.zeros((NRBF+20,NCON)),np.zeros((NRBF+20,NCON)),DX_CON.T))
    B2=np.hstack((np.zeros((NRBF+20,NCON)),PHI_CON.T,np.zeros((NRBF+20,NCON)),DY_CON.T))
    B3=np.hstack((np.zeros((NRBF+20,NCON)),np.zeros((NRBF+20,NCON)),PHI_CON.T,DZ_CON.T))
    B=np.vstack((B1,B2,B3))
    b2=np.hstack((CONu/MAX,CONv/MAX,CONw/MAX,np.zeros(NDIV)))
    return A,B,b1,b2