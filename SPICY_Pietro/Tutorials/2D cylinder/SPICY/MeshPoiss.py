# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:32:05 2022

@author: Pietro.Sperotto
"""
from SPICY.Matrix import Der_RBF_X,Der_RBF_Y,LAP_RBF,Der_RBF_X3D,Der_RBF_Y3D,Der_RBF_Z3D,LAP_RBF3D,Der_RBFXX3D,Der_RBFYY3D,Der_RBFZZ3D,Der_RBF_XZ3D,Der_RBF_YZ3D,Der_RBF_XY3D
import numpy as np
from scipy import linalg
from scipy.sparse import linalg as LA
def Poisson_solver(rho,mu,X_C,Y_C,XG,YG,c,W_u,W_v,X_C_vel,Y_C_vel,cvel,MAT_CON,BC,rcond=1e-13,method='fullcho'):
    """
   Resolve Poisson equation for laminar flow
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure    
                    
     :param XG: array,Grid point of the pressure
     
     :param YG: array,Grid point of the pressure
     
     :param c: array,Shape parameter pressure
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param MAT_CON: array,Matrix of condition
     
     :param BC: array,Boundary conditions
     
     :param rcond: float (optional)
                     inverse of the maximum conditioning acceptable
                    
        
     :param method: string (optional) default is 'fullcho' , other option are 'mixed','fullpinv' and 'solve'
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return W_P: array Pressure weight
   """ 
   #%% forcing term calculation
    #the derivatives of velocity are evaluated
    DER_X=Der_RBF_X(X_C_vel,Y_C_vel,XG,YG,cvel,rcond)
    UX=DER_X.dot(W_u)
    VX=DER_X.dot(W_v)
    del DER_X
    
    #the derivatives of velocity are evaluated
    DER_Y=Der_RBF_Y(X_C_vel,Y_C_vel,XG,YG,cvel,rcond)
    UY=DER_Y.dot(W_u)
    VY=DER_Y.dot(W_v)
    del DER_Y
    
    #forcing term is evaluated
    f=-rho*(UX**2+2*UY*VX+VY**2)
    fmax=np.amax(np.abs(f))#scaler
    
    #laplacian matrix
    L=LAP_RBF(X_C,Y_C,XG,YG,c,rcond)
    
    #Calculation of the whole matrix
    A=2*L.T.dot(L)
    b1=2*L.T.dot(f/fmax)
    del L
    
    #Assigning the boundary condition
    B=MAT_CON.T
    del MAT_CON
    b2=BC/fmax
    
    #%% Solver
    #Solve method directly solve the complete matrix without using Schur complement approach
    if method=='solve':
                W_P=np.linalg.solve(np.vstack((np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))),np.hstack((b1,b2)))
                w=W_P[:len(b1):]
        
    #fullcho method applied a regularization on both A and M (see the article for more details)
    #then both the matrices are solved by using cholesky
    if method=='fullcho':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                A=A+alfa*np.eye(np.shape(A)[0])
                U1=np.linalg.cholesky(A) 
                U1=U1.T
                del A
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                del y
                N=N.T
                M=N.dot(B)
                b2star=N.dot(b1)-b2
                del N
                rhoM=np.linalg.norm(M,np.inf)
                beta=rcond*rhoM
                U2=np.linalg.cholesky(M+beta*np.eye(np.shape(M)[0]))
                U2=U2.T
                y=linalg.solve_triangular(U2.T,b2star,lower=True)
                lam=linalg.solve_triangular(U2,y)
                del U2
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
                
        #Solve both A and M using penrose pseudoinverse
    if method=='fullpinv':
                Ainv=np.linalg.pinv(A,rcond,hermitian=True)
                del A
                N=B.T.dot(Ainv)
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                w=Ainv.dot(b1-B.dot(lam))
        
        #Solve both M using penrose pseudoinverse, while A is solved by using
        #Cholesky decomposition and Tikhonov regularization
    if method=='mixed':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                U1=np.linalg.cholesky(A+alfa*np.eye(np.shape(A)[0]))
                del A
                U1=U1.T
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                N=N.T
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
                
    #Solution is extracted and scaled
    W_P=w*fmax#extracting pressure weigth
    return W_P
def Poisson_solver3D(rho,mu,X_C,Y_C,Z_C,XG,YG,ZG,c,W_u,W_v,W_w,X_C_vel,Y_C_vel,Z_C_vel,cvel,MAT_CON,BC,rcond=1e-13,method='fullcho'):
    """
   Resolve Poisson equation for laminar flow
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure    
     
     :param Z_C: array,Collocation point z of the pressure
                    
     :param XG: array,Grid point of the pressure
     
     :param YG: array,Grid point of the pressure
     
     :param ZG: array,Grid point of the pressure
     
     :param c: array,Shape parameter pressure
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param Z_C_vel: array,Collocation point z of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param MAT_CON: array,Matrix of condition
     
     :param BC: array,Boundary conditions
     
     :param alpha: float,(optional) Regularization parameter
     
     :param rcond: float (optional)
                     inverse of the maximum conditioning acceptable
                    
        
     :param method: string (optional) default is 'fullcho' , other option are 'mixed' and 'fullpinv'
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return W_P: array Pressure weight
   """ 
   #%% forcing term calculation
    #the derivatives of velocity are evaluated
    DER_X=Der_RBF_X3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)#Differentation matrix
    UX=DER_X.dot(W_u)
    VX=DER_X.dot(W_v)
    WX=DER_X.dot(W_w)
    del DER_X
    
    #the derivatives of velocity are evaluated
    DER_Y=Der_RBF_Y3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    UY=DER_Y.dot(W_u)
    VY=DER_Y.dot(W_v)
    WY=DER_Y.dot(W_w)
    del DER_Y
    
    #the derivatives of velocity are evaluated
    DER_Z=Der_RBF_Z3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    UZ=DER_Z.dot(W_u)
    VZ=DER_Z.dot(W_v)
    WZ=DER_Z.dot(W_w)
    del DER_Z
    
    #forcing term is evaluated
    f=-rho*(UX**2+VY**2+WZ**2+2*UY*VX+2*UZ*WX+2*WY*VZ)
    fnorm=np.linalg.norm(f)#scaler
    if fnorm==0:
        fnorm=1
    #laplacian matrix
    L=LAP_RBF3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond)
    
    #Calculation of the whole matrix
    A=2*L.T.dot(L)/fnorm
    b1=2*L.T.dot(f)/fnorm
    
    #Assign the boundary conditions
    B=MAT_CON.T
    del MAT_CON
    b2=BC
    #%% Solver
    #Solve method directly solve the complete matrix without using Schur complement approach
    if method=='solve':
                W_P=np.linalg.solve(np.vstack((np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))),np.hstack((b1,b2)))
                w=W_P[:len(b1):]
        
    #fullcho method applied a regularization on both A and M (see the article for more details)
    #then both the matrices are solved by using cholesky
    if method=='fullcho':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                A=A+alfa*np.eye(np.shape(A)[0])
                U1=np.linalg.cholesky(A) 
                U1=U1.T
                del A
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                del y
                N=N.T
                M=N.dot(B)
                b2star=N.dot(b1)-b2
                del N
                rhoM=np.linalg.norm(M,np.inf)
                beta=rcond*rhoM
                U2=np.linalg.cholesky(M+beta*np.eye(np.shape(M)[0]))
                U2=U2.T
                y=linalg.solve_triangular(U2.T,b2star,lower=True)
                lam=linalg.solve_triangular(U2,y)
                del U2
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
                
    #Solve both A and M using penrose pseudoinverse
    if method=='fullpinv':
                Ainv=np.linalg.pinv(A,rcond,hermitian=True)
                del A
                N=B.T.dot(Ainv)
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                w=Ainv.dot(b1-B.dot(lam))
        
    #Solve both M using penrose pseudoinverse, while A is solved by using
    #Cholesky decomposition and Tikhonov regularization
    if method=='mixed':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                U1=np.linalg.cholesky(A+alfa*np.eye(np.shape(A)[0]))
                del A
                U1=U1.T
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                N=N.T
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
    #Solution is extracted and scaled
    W_P=w#extracting pressure weigth
    #Calculating the L2-norm error
    return W_P
def Poisson_solver3DRSA(rho,mu,X_C,Y_C,Z_C,XG,YG,ZG,c,W_u,W_v,W_w,W_RSXX,W_RSXY,W_RSXZ,W_RSYY,W_RSYZ,W_RSZZ,X_C_vel,Y_C_vel,Z_C_vel,cvel,MAT_CON,BC,rcond=1e-13,method='fullcho'):
    """
   Resolve Poisson equation for laminar flow
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure    
     
     :param Z_C: array,Collocation point z of the pressure
                    
     :param XG: array,Grid point of the pressure
     
     :param YG: array,Grid point of the pressure
     
     :param ZG: array,Grid point of the pressure
     
     :param c: array,Shape parameter pressure
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param Z_C_vel: array,Collocation point z of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param MAT_CON: array,Matrix of condition
     
     :param BC: array,Boundary conditions
     
     :param alpha: float,(optional) Regularization parameter
     
     :param rcond: float (optional)
                     inverse of the maximum conditioning acceptable
                    
        
     :param method: string (optional) default is 'fullcho' , other option are 'mixed' and 'fullpinv'
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return W_P: array Pressure weight
   """ 
   #%% forcing term calculation
    #the derivatives of velocity are evaluated
    DER_X=Der_RBF_X3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)#Differentation matrix
    UX=DER_X.dot(W_u)
    VX=DER_X.dot(W_v)
    WX=DER_X.dot(W_w)
    del DER_X
    
    #the derivatives of velocity are evaluated
    DER_Y=Der_RBF_Y3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    UY=DER_Y.dot(W_u)
    VY=DER_Y.dot(W_v)
    WY=DER_Y.dot(W_w)
    del DER_Y
    
    #the derivatives of velocity are evaluated
    DER_Z=Der_RBF_Z3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    UZ=DER_Z.dot(W_u)
    VZ=DER_Z.dot(W_v)
    WZ=DER_Z.dot(W_w)
    del DER_Z
    DER_XX=Der_RBFXX3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    DXXRS=DER_XX.dot(W_RSXX)
    del DER_XX
    DER_YY=Der_RBFYY3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    DYYRS=DER_YY.dot(W_RSYY)
    del DER_YY
    DER_ZZ=Der_RBFZZ3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    DZZRS=DER_ZZ.dot(W_RSZZ)
    del DER_ZZ
    DER_XZ=Der_RBF_XZ3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    DXZRS=DER_XZ.dot(W_RSXZ)
    del DER_XZ
    DER_YZ=Der_RBF_YZ3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    DYZRS=DER_YZ.dot(W_RSYZ)
    del DER_YZ    
    DER_XY=Der_RBF_XY3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    DXYRS=DER_XY.dot(W_RSXY)
    del DER_XY   
    #forcing term is evaluated
    f=-rho*(UX**2+VY**2+WZ**2+2*UY*VX+2*UZ*WX+2*WY*VZ+DXXRS+DYYRS+DZZRS+2*DXZRS+2*DYZRS+2*DXYRS)
    fmax=np.amax(np.abs(f))#scaler
    #laplacian matrix
    L=LAP_RBF3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond)
    
    #Calculation of the whole matrix
    A=2*L.T.dot(L)
    b1=2*L.T.dot(f)/fmax
    
    #Assign the boundary conditions
    B=MAT_CON.T
    del MAT_CON
    b2=BC/fmax
    #%% Solver
    #Solve method directly solve the complete matrix without using Schur complement approach
    if method=='solve':
                AA=np.vstack(np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))
                del A,B
                W_P=np.linalg.solve(AA,np.hstack((b1,b2)),tol=1e-7)
                w=W_P[:len(b1):]
        
    #fullcho method applied a regularization on both A and M (see the article for more details)
    #then both the matrices are solved by using cholesky
    if method=='fullcho':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                A=A+alfa*np.eye(np.shape(A)[0])
                U1=np.linalg.cholesky(A) 
                U1=U1.T
                del A
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                del y
                N=N.T
                M=N.dot(B)
                b2star=N.dot(b1)-b2
                del N
                rhoM=np.linalg.norm(M,np.inf)
                beta=rcond*rhoM
                U2=np.linalg.cholesky(M+beta*np.eye(np.shape(M)[0]))
                U2=U2.T
                y=linalg.solve_triangular(U2.T,b2star,lower=True)
                lam=linalg.solve_triangular(U2,y)
                del U2
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
                
    #Solve both A and M using penrose pseudoinverse
    if method=='fullpinv':
                Ainv=np.linalg.pinv(A,rcond,hermitian=True)
                del A
                N=B.T.dot(Ainv)
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                w=Ainv.dot(b1-B.dot(lam))
        
    #Solve both M using penrose pseudoinverse, while A is solved by using
    #Cholesky decomposition and Tikhonov regularization
    if method=='mixed':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                U1=np.linalg.cholesky(A+alfa*np.eye(np.shape(A)[0]))
                del A
                U1=U1.T
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                N=N.T
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
    #Solution is extracted and scaled
    #Calculating the L2-norm error
    return w*fmax
def Poisson_solver3Dml(rho,mu,X_C,Y_C,Z_C,XG,YG,ZG,c,W_u,W_v,W_w,W_RSI,X_C_vel,Y_C_vel,Z_C_vel,cvel,MAT_CON,BC,X_C_old,Y_C_old,Z_C_old,c_P_old,w_P_old,rcond=1e-13,method='fullcho'):
    """
   Resolve Poisson equation for laminar flow
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure    
     
     :param Z_C: array,Collocation point z of the pressure
                    
     :param XG: array,Grid point of the pressure
     
     :param YG: array,Grid point of the pressure
     
     :param ZG: array,Grid point of the pressure
     
     :param c: array,Shape parameter pressure
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param Z_C_vel: array,Collocation point z of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param MAT_CON: array,Matrix of condition
     
     :param BC: array,Boundary conditions
     
     :param alpha: float,(optional) Regularization parameter
     
     :param rcond: float (optional)
                     inverse of the maximum conditioning acceptable
                    
        
     :param method: string (optional) default is 'fullcho' , other option are 'mixed' and 'fullpinv'
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return W_P: array Pressure weight
   """ 
   #%% forcing term calculation
    #the derivatives of velocity are evaluated
    DER_X=Der_RBF_X3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)#Differentation matrix
    UX=DER_X.dot(W_u)
    VX=DER_X.dot(W_v)
    WX=DER_X.dot(W_w)
    del DER_X
    
    #the derivatives of velocity are evaluated
    DER_Y=Der_RBF_Y3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    UY=DER_Y.dot(W_u)
    VY=DER_Y.dot(W_v)
    WY=DER_Y.dot(W_w)
    del DER_Y
    
    #the derivatives of velocity are evaluated
    DER_Z=Der_RBF_Z3D(X_C_vel,Y_C_vel,Z_C_vel,XG,YG,ZG,cvel,rcond)
    UZ=DER_Z.dot(W_u)
    VZ=DER_Z.dot(W_v)
    WZ=DER_Z.dot(W_w)
    del DER_Z
    
    #Second part of reynolds stresses is computed
    #forcing term is evaluated
    L=LAP_RBF3D(X_C_old,Y_C_old,Z_C_old,XG,YG,ZG,c_P_old,rcond)
    f=-rho*(UX**2+VY**2+WZ**2+2*UY*VX+2*UZ*WX+2*WY*VZ)-L@w_P_old
    del L
    fnorm=np.linalg.norm(f)#scaler
    #laplacian matrix
    L=LAP_RBF3D(X_C,Y_C,Z_C,XG,YG,ZG,c,rcond)
    
    #Calculation of the whole matrix
    A=2*L.T.dot(L)/fnorm
    b1=2*L.T.dot(f/fnorm)
    
    #Assign the boundary conditions
    B=MAT_CON.T
    del MAT_CON
    b2=BC
    #%% Solver
    #Solve method directly solve the complete matrix without using Schur complement approach
    if method=='solve':
                W_P=np.linalg.solve(np.vstack((np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))),np.hstack((b1,b2)))
                w=W_P[:len(b1):]
    if method=='minres':
                W_P,_=LA.minres(np.vstack((np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))),np.hstack((b1,b2)),tol=1e-20)
                w=W_P[:len(b1):]
        
    #fullcho method applied a regularization on both A and M (see the article for more details)
    #then both the matrices are solved by using cholesky
    if method=='fullcho':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                A=A+alfa*np.eye(np.shape(A)[0])
                U1=np.linalg.cholesky(A) 
                U1=U1.T
                del A
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                del y
                N=N.T
                M=N.dot(B)
                b2star=N.dot(b1)-b2
                del N
                rhoM=np.linalg.norm(M,np.inf)
                beta=rcond*rhoM
                U2=np.linalg.cholesky(M+beta*np.eye(np.shape(M)[0]))
                U2=U2.T
                y=linalg.solve_triangular(U2.T,b2star,lower=True)
                lam=linalg.solve_triangular(U2,y)
                del U2
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
                
    #Solve both A and M using penrose pseudoinverse
    if method=='fullpinv':
                Ainv=np.linalg.pinv(A,rcond,hermitian=True)
                del A
                N=B.T.dot(Ainv)
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                w=Ainv.dot(b1-B.dot(lam))
        
    #Solve both M using penrose pseudoinverse, while A is solved by using
    #Cholesky decomposition and Tikhonov regularization
    if method=='mixed':
                rhoA=np.linalg.norm(A,np.inf)
                alfa=rcond*rhoA
                U1=np.linalg.cholesky(A+alfa*np.eye(np.shape(A)[0]))
                del A
                U1=U1.T
                y=linalg.solve_triangular(U1.T,B,lower=True)
                N=linalg.solve_triangular(U1,y)
                N=N.T
                M=N.dot(B)
                Minv=np.linalg.pinv(M,rcond,hermitian=True)
                del M
                lam=Minv.dot(N.dot(b1)-b2)
                y=linalg.solve_triangular(U1.T,b1-B.dot(lam),lower=True)
                w=linalg.solve_triangular(U1,y)
    #Solution is extracted and scaled

    W_P=w#extracting pressure weigth
    #Calculating the L2-norm error
    return W_P,np.linalg.norm(f-L@W_P)/fnorm,np.linalg.norm(BC-B.T@W_P)/np.linalg.norm(BC)