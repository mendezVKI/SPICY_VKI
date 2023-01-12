# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:09:22 2022

@author: Pietro.Sperotto
"""
from scipy import linalg
import numpy as np
from SPICY.Matrix import PHI,Der_RBF_X,Der_RBF_Y,PHI3D,Der_RBF_X3D,Der_RBF_Y3D,Der_RBF_Z3D
def Inter_2D_C(u,v,XG,YG,X_C,Y_C,c_V,XCON,YCON,
               Constr_u,Constr_v,XDIV=np.array([]),YDIV=np.array([]),rcond=1e-13,DIV=0,method='fullcho',constraint_elimination=False):
        """
        This method computes the weights Wu Wv for the velocity
        field (u,v) sampled over a grid (Xg,Yg), using collocation points
        (X_C,Y_C). 
             
        
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param u: array 
                    One dimensional array of the velocity  in the x direction to be fitted

        :param v: array 
                    One dimensional array of the velocity  in the y direction to be fitted
                    
        :param XG: array 
                    One dimensional array of the X position of u and v
                    
        :param YG: array 
                    One dimensional array of the Y position of u and v   
                    
        :param X_C: array 
                    One dimensional array of the X position of the collocation points
                    
        :param Y_C: array 
                    One dimensional array of the Y position of the collocation points
                    
        :param c_V: array 
                    Shape parameter of every gaussians
                    
        :param XCON: array 
                    X Positions of constraints
                    
        :param YCON: array 
                    Y Positions of constraints
                    
        :param Constr_u: array 
                    Constraint for the velocity u in XCON and YCON  
                    
        :param Constr_v: array 
                    Constraint for the velocity v in XCON and YCON
                    
        :param XDIV: array (optional)
                    X position where impose the divergence free without imposing a constraint
                    
        :param YDIV: array (optional)
                    Y position where impose the divergence free without imposing a constraint
                    
        :param alpha: float (optional)
                    Regularization parameter, if it is set the whole system is solved by using numpy.solve()
                    
        :param rcond: float (optional)
                    inverse of the maximum conditioning acceptable
                    
        :param DIV: float (optional) 
                    0 if no Divergence penalty 1 if there is
        
        :param method: string (optional) 
                    default is 'fullcho' , other option are 'mixed' and 'fullpinv'. 'fullcho' uses only Cholensky.
                    'fullpinv' uses only pseudo inverse. 'mixed' uses Cholensky for the variance matrix
                    and a pseudoinverse for the constraint part. Finally, 'solve' use np.linalg.solve on the complete matrix.
        :param constraint_elimination: bool (optional)
                                       enables constraint elimination through QR decomposition, default is False
        ----------------------------------------------------------------------------------------------------------------
        Returns
        -------

        :return: W_u array 
                Array of the weights of the RBF functions for  the u velocity
        :return: W_v array 
                Array of the weights of the RBF functions for  the v velocity
        """
        #%% Creation of the matrix
        NRBF=len(X_C)#number of RBF used
        NCON=len(XCON)#number of data points
        #This next if check if the divergence-free penalty is required by user or no
        if XDIV.size==0:
            
        #If the condition is respected then there are no pure divergence-free constraint
        #Then the differentiation matrix for the constraint has to be evaluated only in XCON and YCON
         vmax=np.amax(np.abs(v))
         umax=np.amax(np.abs(u))
         MAX=max((umax,vmax))
         
         #Create the matrix PHI referred to the data points
         PHI_X=(PHI(X_C,Y_C,XG,YG,c_V,rcond))       
         b1=np.hstack((2*PHI_X.T.dot(u),2*PHI_X.T.dot(v)))/MAX
         PHITPHI=2*PHI_X.T.dot(PHI_X)
         del PHI_X
         
         #Create the differentiations matrices DX,DY referred to the data points
         DX=Der_RBF_X(X_C,Y_C,XG,YG,c_V,rcond)
         DY=Der_RBF_Y(X_C,Y_C,XG,YG,c_V,rcond)
         DXTDY=2*DX.T.dot(DY)
         
         #Compute the matrices temporary A1,A2
         A1=np.hstack((PHITPHI+2*DIV*DX.T.dot(DX),DIV*DXTDY))
         A2=np.hstack((DIV*DXTDY.T,PHITPHI+DIV*2*DY.T.dot(DY)))
         del PHITPHI,DXTDY
         
         #Stack the variance matrix
         A=np.vstack((A1,A2))
         del A1,A2    
         
         #Compute the DX,DY and PHI in the Boundary
         DX_CON=Der_RBF_X(X_C,Y_C,XCON,YCON,c_V,rcond)
         PHI_CON=PHI(X_C,Y_C,XCON,YCON,c_V,rcond)
         B1=np.hstack((PHI_CON.T,np.zeros((NRBF+10,NCON)),DX_CON.T))
         del DX_CON
         DY_CON=Der_RBF_Y(X_C,Y_C,XCON,YCON,c_V,rcond)
         B2=np.hstack((np.zeros((NRBF+10,NCON)),PHI_CON.T,DY_CON.T))
         del DY_CON,PHI_CON
         
         #Compute the final constraint matrix
         B=np.vstack((B1,B2))
         del B1,B2    
         
         #Compute the constraints
         b2=np.hstack((Constr_u,Constr_v,np.zeros(NCON)))/MAX
        else:
         if XCON.size==0:
             #If the condition is respected then there are no velocity constraint
             #Then the constraint matrix is evaluated only at XDIV,YDIV
             vmax=np.amax(np.abs(v))
             umax=np.amax(np.abs(u))
             MAX=max((umax,vmax))
             
             #Create the matrix PHI referred to the data points       
             PHI_X=(PHI(X_C,Y_C,XG,YG,c_V,rcond))
             b1=np.hstack((2*PHI_X.T.dot(u),2*PHI_X.T.dot(v)))/MAX
             PHITPHI=2*PHI_X.T.dot(PHI_X)
             del PHI_X
             
             #Create the differentiations matrices DX,DY referred to the data points
             DX=Der_RBF_X(X_C,Y_C,XG,YG,c_V,rcond)
             DY=Der_RBF_Y(X_C,Y_C,XG,YG,c_V,rcond)
             DXTDY=2*DX.T.dot(DY)
             
             #Compute the matrices temporary A1,A2
             A1=np.hstack((PHITPHI+2*DIV*DX.T.dot(DX),DIV*DXTDY))
             A2=np.hstack((DIV*DXTDY.T,PHITPHI+DIV*2*DY.T.dot(DY)))
             del PHITPHI
             
             #Stack the variance matrix
             A=np.vstack((A1,A2))
             del A1,A2
             
             #Compute the DX,DY in the Boundary
             DX_CON=Der_RBF_X(X_C,Y_C,XDIV,YDIV,c_V,rcond)
             DY_CON=Der_RBF_Y(X_C,Y_C,XDIV,YDIV,c_V,rcond)
             
             #Compute the final constraint matrix
             B=np.vstack((DX_CON.T,DY_CON.T))
             del DX_CON,DY_CON
             
             #Compute the constraints
             b2=np.zeros(len(XDIV))
             
         else:
             #If the code enters here means that there are both pure divergence-free constraint and velocity constraint
             
             #The two set of point of constraint, indeed the divergence-free constraint is applied in both XCON and XDIV
             XT=np.hstack((XCON,XDIV))
             YT=np.hstack((YCON,YDIV))
             #XCON and YCON are stacked with XDIV and YDIV  and the differentiation matrices of the constraint are evaluated on these point
             #NB only the differentiation matrix are evaluated on XT and YT the others as PHI_CON are still evaluated on XCON and YCON
             
             NDIV=len(XT)
             vmax=np.amax(np.abs(v))
             umax=np.amax(np.abs(u))
             MAX=max((umax,vmax))
             
             #Create the matrix PHI referred to the data points 
             PHI_X=PHI(X_C,Y_C,XG,YG,c_V,rcond)
             b1=np.hstack((2*PHI_X.T.dot(u),2*PHI_X.T.dot(v)))/MAX
             PHITPHI=2*PHI_X.T.dot(PHI_X)
             del PHI_X
             
             #Create the differentiations matrices DX,DY referred to the data points
             DX=Der_RBF_X(X_C,Y_C,XG,YG,c_V,rcond)
             DY=Der_RBF_Y(X_C,Y_C,XG,YG,c_V,rcond)
             DXTDY=2*DX.T.dot(DY)
             
             #Compute the matrices temporary A1,A2
             A1=np.hstack((PHITPHI+DIV*2*DX.T.dot(DX),DIV*DXTDY))
             del DX
             A2=np.hstack((DIV*DXTDY.T,PHITPHI+DIV*2*DY.T.dot(DY)))
             del DY,PHITPHI
             
             #Stack the variance matrix
             A=np.vstack((A1,A2))
             del A1,A2     
             
             #Compute the DX,DY,PHI in the Boundary
             DX_CON=Der_RBF_X(X_C,Y_C,XT,YT,c_V,rcond)
             PHI_CON=PHI(X_C,Y_C,XCON,YCON,c_V,rcond)
             B1=np.hstack((PHI_CON.T,np.zeros((NRBF+10,NCON)),DX_CON.T))
             del DX_CON
             DY_CON=Der_RBF_Y(X_C,Y_C,XT,YT,c_V,rcond)
             B2=np.hstack((np.zeros((NRBF+10,NCON)),PHI_CON.T,DY_CON.T))
             del DY_CON,PHI_CON
             
             #Compute the final constraint matrix
             B=np.vstack((B1,B2))
             del B1,B2    
             b2=np.hstack((Constr_u,Constr_v,np.zeros(NDIV)))/MAX
        
        #This eliminates the redundant constraints
        #was just a prove and does not seem to work properly therefore is deactivated
        #But I keep if needed to be rechecked
        if constraint_elimination:
            Q,R,P=linalg.qr(B.T,check_finite=False,pivoting=True)
            b2=Q.T.dot(b2)
            Rii=np.diag(R)
            Logical=np.abs(Rii)>np.amax(np.abs(Rii))*rcond
            B=R[Logical,:]
            B=B.T

            b2=b2[Logical]
        #%% Solver
        #Solve method directly solve the complete matrix without using Schur complement approach
        if method=='solve':
                W_vel=np.linalg.solve(np.vstack((np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))),np.hstack((b1,b2)))
                w=W_vel[:len(b1):]
        
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
        #W_u and W_v are obtained and multiplied for its scaler
        W_u=w[0:NRBF+10]*MAX
        W_v=w[NRBF+10:2*(NRBF+10)]*MAX

        return W_u,W_v
def Inter_3D_C(u,v,w,XG,YG,ZG,X_C,Y_C,Z_C,c_V,XCON,YCON,ZCON,
                   Constr_u,Constr_v,Constr_w,XDIV=np.array([]),YDIV=np.array([]),ZDIV=np.array([]),rcond=1e-13,DIV=0,method='fullcho'):
            """
            This method computes the weights Wu Wv for the velocity
            field (u,v) sampled over a grid (Xg,Yg), using collocation points
            (X_C,Y_C). 
                 
            
            ----------------------------------------------------------------------------------------------------------------
            Parameters
            ----------
            :param u: array 
                        One dimensional array of the velocity  in the x direction to be fitted

            :param v: array 
                        One dimensional array of the velocity  in the y direction to be fitted
                        
            :param w: array 
                        One dimensional array of the velocity  in the z direction to be fitted
                        
            :param XG: array 
                        One dimensional array of the X position of u,v,w
                        
            :param YG: array 
                        One dimensional array of the Y position of u,v,w   
                        
            :param YG: array 
                        One dimensional array of the Z position of u,v,w  
                        
            :param X_C: array 
                        One dimensional array of the X position of the collocation points
                        
            :param Y_C: array 
                        One dimensional array of the Y position of the collocation points

            :param Z_C: array 
                        One dimensional array of the Z position of the collocation points
                        
            :param c_V: array 
                        Shape parameter of every gaussians
                        
            :param XCON: array 
                        X Positions of constraints
                        
            :param YCON: array 
                        Y Positions of constraints
                        
            :param ZCON: array 
                        Z Positions of constraints
                        
            :param Constr_u: array 
                        Constraint for the velocity u in XCON,YCON  and ZCON
                        
            :param Constr_v: array 
                        Constraint for the velocity v in XCON,YCON  and ZCON
                        
            :param Constr_w: array 
                        Constraint for the velocity w in XCON,YCON  and ZCON
                        
            :param XDIV: array (optional)
                        X position where impose the divergence free without imposing a constraint
                        
            :param YDIV: array (optional)
                        Y position where impose the divergence free without imposing a constraint
                        
            :param ZDIV: array (optional)
                        Z position where impose the divergence free without imposing a constraint
                        
            :param rcond: float (optional)
                        inverse of the maximum conditioning acceptable
                        
            :param DIV: float (optional) 
                        0 if no Divergence penalty 1 if there is
            
            :param method: string (optional) 
                        default is 'fullcho' , other option are 'mixed' and 'fullpinv'. 'fullcho' uses only Cholensky
                        'fullpinv' uses only pseudo inverse and finally 'mixed' uses Cholensky for the variance matrix
                        and a pseudoinverse for the constraint part. Finally, 'solve' use np.linalg.solve on the complete matrix.
            
            ----------------------------------------------------------------------------------------------------------------
            Returns
            -------

            :return: W_u array 
                    Array of the weights of the RBF functions for  the u velocity
            :return: W_v array 
                    Array of the weights of the RBF functions for  the v velocity
            """
            #%% Creation of the matrix
            NRBF=len(X_C)#number of RBF used
            NCON=len(XCON)#number of data points
            
            #This next if check if the divergence-free penalty is required by user or not
            #This check if there are only divergence-free constraint
            if XDIV.size==0:
            #If the condition are respected then there are no pure divergence-free constraint
            #Then the differentiation matrix for the constraint has to evaluated in XCON and YCON
               unorm=(np.linalg.norm(u))**2
               vnorm=(np.linalg.norm(v))**2
               wnorm=(np.linalg.norm(w))**2
               
               #Create the matrix PHI referred to the data points 
               PHI=PHI3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond)
               b1=np.hstack((2*PHI.T.dot(u/(unorm)),2*PHI.T.dot(v/(vnorm)),2*PHI.T.dot(w/(wnorm))))
               PHITPHI=2*PHI.T.dot(PHI)
               del PHI
             
               #Create the differentiations matrices DX,DY,DZ referred to the data points
               DX=(Der_RBF_X3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
               DY=(Der_RBF_Y3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
               DZ=(Der_RBF_Z3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
               DXTDY=2*DX.T.dot(DY)
               DXTDZ=2*DX.T.dot(DZ)
               DYTDZ=2*DY.T.dot(DZ)
               #Create variance matrix A
               A=np.vstack((np.hstack((PHITPHI/unorm+DIV*2*DX.T.dot(DX),DIV*DXTDY,DIV*DXTDZ)),np.hstack((DIV*DXTDY.T,PHITPHI/vnorm+DIV*2*DY.T.dot(DY),DIV*DYTDZ)),np.hstack((DIV*DXTDZ.T,+DIV*DYTDZ.T,DIV*2*DZ.T.dot(DZ)+PHITPHI/wnorm))))
               del DX,DZ,DY,DXTDY,DXTDZ,DYTDZ,PHITPHI
               
               #Compute the DX,DY,DZ,PHI in the Boundary
               DX_CON=(Der_RBF_X3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond))
               DY_CON=(Der_RBF_Y3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond))
               DZ_CON=(Der_RBF_Z3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond))
               PHI_CON=(PHI3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond))
               
               #Compute the constraint Matrix
               B=np.vstack((np.hstack((PHI_CON.T,np.zeros((NRBF+20,NCON)),np.zeros((NRBF+20,NCON)),DX_CON.T)),np.hstack((np.zeros((NRBF+20,NCON)),PHI_CON.T,np.zeros((NRBF+20,NCON)),DY_CON.T)),np.hstack((np.zeros((NRBF+20,NCON)),np.zeros((NRBF+20,NCON)),PHI_CON.T,DZ_CON.T))))
               del PHI_CON,DX_CON,DY_CON,DZ_CON
               
               #Compute the constraints
               b2=np.hstack((Constr_u,Constr_v,Constr_w,np.zeros(NCON)))
            else:
                
             if XCON.size==0:
               #If the condition is respected then there are no velocity constraint
               #Then the constraint matrix is evaluated only at XDIV,YDIV,ZDIV
               NDIV=len(XDIV)
               unorm=(np.linalg.norm(u))**2
               vnorm=(np.linalg.norm(v))**2
               wnorm=(np.linalg.norm(w))**2
               #Create the matrix PHI referred to the data points 
               PHI=PHI3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond)
               b1=np.hstack((2*PHI.T.dot(u/(unorm)),2*PHI.T.dot(v/(vnorm)),2*PHI.T.dot(w/(wnorm))))
               PHITPHI=2*PHI.T.dot(PHI)
               del PHI
               
               #Create the differentiations matrices DX,DY,DZ referred to the data points
               DX=(Der_RBF_X3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
               DY=(Der_RBF_Y3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
               DZ=(Der_RBF_Z3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
               DXTDY=2*DX.T.dot(DY)
               DXTDZ=2*DX.T.dot(DZ)
               DYTDZ=2*DY.T.dot(DZ)
               
               #Create variance matrix A
               A=np.vstack((np.hstack((PHITPHI/unorm+DIV*2*DX.T.dot(DX),DIV*DXTDY,DIV*DXTDZ)),np.hstack((DIV*DXTDY.T,PHITPHI/vnorm+DIV*2*DY.T.dot(DY),DIV*DYTDZ)),np.hstack((DIV*DXTDZ.T,+DIV*DYTDZ.T,DIV*2*DZ.T.dot(DZ)+PHITPHI/wnorm))))
               del DX,DZ,DY,DXTDY,DXTDZ,DYTDZ,PHITPHI
               
               #Compute the DX,DY,DZ in the Boundary
               DX_CON=(Der_RBF_X3D(X_C,Y_C,Z_C,XDIV,YDIV,ZDIV,c_V,rcond))
               DY_CON=(Der_RBF_Y3D(X_C,Y_C,Z_C,XDIV,YDIV,ZDIV,c_V,rcond))
               DZ_CON=(Der_RBF_Z3D(X_C,Y_C,Z_C,XDIV,YDIV,ZDIV,c_V,rcond))
               
               #Compute the constraint matrix
               B=np.vstack((DX_CON.T,DY_CON.T,DZ_CON.T))
               del DX_CON,DY_CON,DZ_CON
               
               #Compute the constraints
               b2=np.hstack((np.zeros(NDIV)))
             else:
                 #If the code enters here means that there are both pure divergence-free constraint and velocity constraint
                 #The two set of point of constraint, indeed the divergence-free constraint is applied in both XCON and XDIV
                 #Otherwise there are some only divergence-free
                 XT=np.hstack((XCON,XDIV))
                 YT=np.hstack((YCON,YDIV))
                 ZT=np.hstack((ZCON,ZDIV))
                 #XCON and YCON are stacked with XDIV and YDIV  and the differentiation matrices of the constraint are evaluated on these point
                 #NB only the differentiation matrix are evaluated on XT,YT and ZT the others as PHI_CON are still evaluated on XCON,YCON and ZCON
                 NDIV=len(XT)
                 unorm=(np.linalg.norm(u))**2
                 vnorm=(np.linalg.norm(v))**2
                 wnorm=(np.linalg.norm(w))**2     
                 #Create the matrix PHI referred to the data points 
                 PHI=PHI3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond)
                 b1=np.hstack((2*PHI.T.dot(u),2*PHI.T.dot(v),2*PHI.T.dot(w)))
                 PHITPHI=2*PHI.T.dot(PHI)
                 del PHI
                 
                 #Create the differentiations matrices DX,DY,DZ referred to the data points
                 DX=(Der_RBF_X3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
                 DY=(Der_RBF_Y3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
                 DZ=(Der_RBF_Z3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond))
                 DXTDY=2*DX.T.dot(DY)
                 DXTDZ=2*DX.T.dot(DZ)
                 DYTDZ=2*DY.T.dot(DZ)
                 #Create variance matrix A
                 A=np.vstack((np.hstack((PHITPHI+DIV*2*DX.T.dot(DX),DIV*DXTDY,DIV*DXTDZ)),np.hstack((DIV*DXTDY.T,PHITPHI+DIV*2*DY.T.dot(DY),DIV*DYTDZ)),np.hstack((DIV*DXTDZ.T,+DIV*DYTDZ.T,DIV*2*DZ.T.dot(DZ)+PHITPHI))))
                 del DX,DZ,DY,DXTDY,DXTDZ,DYTDZ,PHITPHI
                 
                 #Compute the DX,DY,DZ in the Boundary
                 DX_CON=(Der_RBF_X3D(X_C,Y_C,Z_C,XT,YT,ZT,c_V,rcond))
                 DY_CON=(Der_RBF_Y3D(X_C,Y_C,Z_C,XT,YT,ZT,c_V,rcond))
                 DZ_CON=(Der_RBF_Z3D(X_C,Y_C,Z_C,XT,YT,ZT,c_V,rcond))
                 PHI_CON=(PHI3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond))
                 
                 #Compute the constraint matrix
                 B=np.vstack((np.hstack((PHI_CON.T,np.zeros((NRBF+20,NCON)),np.zeros((NRBF+20,NCON)),DX_CON.T)),np.hstack((np.zeros((NRBF+20,NCON)),PHI_CON.T,np.zeros((NRBF+20,NCON)),DY_CON.T)),np.hstack((np.zeros((NRBF+20,NCON)),np.zeros((NRBF+20,NCON)),PHI_CON.T,DZ_CON.T))))
                 del PHI_CON,DX_CON,DY_CON,DZ_CON
                 
                 #Compute the constraints
                 b2=np.hstack((Constr_u,Constr_v,Constr_w,np.zeros(NDIV)))
            #%% Solution of the matrix
            #If alpha is zero then one of the "method" is used otherwise numpy.solve() is used
                #The different methods applies for reference look in the article
            if method=='solve':
                W_vel=np.linalg.solve(np.vstack((np.hstack((A,B)),np.hstack((B.T,np.zeros((len(b2),len(b2))))))),np.hstack((b1,b2)))
                
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
                    W_vel=np.hstack((w,lam))
            
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
                    W_vel=np.hstack((w,lam))
                    
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
                    W_vel=np.hstack((w,lam))
                    
            #W_u and W_v are obtained and multiplied for its scaler

            W_u=W_vel[0:NRBF+20]
            W_v=W_vel[NRBF+20:2*(NRBF+20)]
            W_w=W_vel[2*(NRBF+20):3*(NRBF+20)]

            return W_u,W_v,W_w
def Inter_3D_RSI(RS,XG,YG,ZG,X_C,Y_C,Z_C,c_V,XCON,YCON,ZCON,CON,rcond=1e-13,method='fullcho'):
   if len(XCON)==0:
    #standard A
    PHI=PHI3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond)
    A=2*PHI.T@(PHI)
    maxRS=np.amax(RS,axis=0)
    if isinstance(maxRS,np.ndarray):
        maxRS[maxRS==0]=1   
    else:
        if maxRS==0:
            maxRS=1
    #the solver is the same
    if method=='fullcho' or method=='mixed':
       rhoA=np.linalg.norm(A,np.inf)
       alfa=rcond*rhoA
       U1=np.linalg.cholesky(A+alfa*np.eye(np.shape(A)[0]))
       del A
       y=linalg.solve_triangular(U1,2*PHI.T@(RS/(maxRS)),lower=True)
       w=linalg.solve_triangular(U1.T,y)
    if method=='fullpinv':
                Ainv=np.linalg.pinv(A,rcond,hermitian=True)
                w=Ainv.dot(2*PHI.T@(RS/(np.amax(RS,axis=0))))
   else:
    #standard A
    PHI=PHI3D(X_C,Y_C,Z_C,XG,YG,ZG,c_V,rcond)
    A=2*PHI.T@(PHI)
    maxRS=np.amax(RS,axis=0)
    maxRS[maxRS==0]=1
    b1=2*PHI.T@(RS/maxRS)
    del PHI
    B=np.vstack((PHI3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond),Der_RBF_X3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond),Der_RBF_Y3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond),Der_RBF_Z3D(X_C,Y_C,Z_C,XCON,YCON,ZCON,c_V,rcond))).T
    b2=np.hstack((CON,np.zeros(3*len(XCON))))
    if np.size(np.shape(RS))!=1:
        b2temp=np.zeros((len(b2),np.shape(RS)[1]))
        for k in np.arange(np.shape(RS)[1]):
            b2temp[:,k]=b2
    del CON
    b2=np.copy(b2temp)
    del b2temp
    #the solver is the same
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
   return w*maxRS