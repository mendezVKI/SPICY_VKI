# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:28:37 2022

@author: Pietro.Sperotto
"""
import numpy as np
from SPICY.Matrix import PHI,Der_RBF_X,Der_RBF_Y,LAP_RBF,PHI3D,Der_RBF_X3D,Der_RBF_Y3D,Der_RBF_Z3D,LAP_RBF3D
def Boundary_Conditions(rho,mu,XBC,YBC,BCD,n,W_u,W_v,X_C_vel,Y_C_vel,cvel,X_C,Y_C,c,rcond):
    """
   This function calculates the Boundary conditions and the condition matrix for the problem
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
                        
     :param XBC: tuple, It contains in its cells the one dimensional array of the point in X of every
                 boundary
                        
     :param YBC: tuple, It contains in its cells the one dimensional array of the point in Y of every
                 boundary
                        
     :param BCD: tuple, It contains only the Dirichlet conditions , in the cells of the tuple corresponding
                to a Neumann condition instead of an array with the conditions simply a string 'Neumann'
                has to be added in order to perform better in the Neumann condition is apply to a wall
                is preferable insert 'Neumann Wall'
     
     :param n: tuple, contains the normal normalized vector evaluated in the corresponding point of XBC and YBC for every edge which has Neumann condition
               If Dirichlet condition are present in its cell can be every but still something has to be put
               The 2D matrix to insert in every cell must have shape 2XN where N is the lenght of the equivalent cell of XBC
               the first row is the projection of the normal vector in x the second in y
               To remember the normal vector norm in any point has to be one 
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure
     
     :param c: array,Shape parameter pressure
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return BC: array boundary condition  for every point
     :return MAT_CON: array, matrix of constraint
   """ 
   #%% Loop to create the boundary conditions in every edge
    for k in np.arange(0,len(XBC)):
        #if the variable BC is a string therefore the condition is Neumann or Neumann Wall
        if isinstance(BCD[k], str):
           #Create differentation matrices needed to construct the momentum equation
           
           #Evaluate the laplacian fo both velocities
           LAP_vel=LAP_RBF(X_C_vel,Y_C_vel,XBC[k],YBC[k],cvel,rcond)
           LAPU=LAP_vel.dot(W_u)
           LAPV=LAP_vel.dot(W_v)
           del LAP_vel

           #copy the normals to avoid damaging it
           nn=np.copy(n[k])
           #Extract the normal direction for x and y
           nx=nn[0,:]
           ny=nn[1,:]
           #%% Neumann conditions
           if BCD[k]=='Neumann':
              #Evaluate the X derivatives 
              DX_vel=Der_RBF_X(X_C_vel,Y_C_vel,XBC[k],YBC[k],cvel,rcond)
              UX=DX_vel.dot(W_u)
              VX=DX_vel.dot(W_v)
              del DX_vel
              
              #Evaluate the Y derivatives
              DY_vel=Der_RBF_Y(X_C_vel,Y_C_vel,XBC[k],YBC[k],cvel,rcond)
              UY=DY_vel.dot(W_u)
              VY=DY_vel.dot(W_v)
              del DY_vel
              
              #Evaluate the velocities
              PHI_vel=PHI(X_C_vel,Y_C_vel,XBC[k],YBC[k],cvel,rcond)
              U=PHI_vel.dot(W_u)
              V=PHI_vel.dot(W_v)
              del PHI_vel
              
              #dP/dx and dP/dy from Navier-Stokes
              BCx=-rho*(U*UX+V*UY)+mu*LAPU
              BCy=-rho*(U*VX+V*VY)+mu*LAPV
              
              #The gradient of the pressure is projected on 
              #the normal direction to the boundary
              BCk=BCx*nx+BCy*ny
              del BCx,BCy
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X(X_C,Y_C,XBC[k],YBC[k],c,rcond)
              DY_P=Der_RBF_Y(X_C,Y_C,XBC[k],YBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T
              del DY_P,DX_P
              
           if BCD[k]=='Neumann Wall':
              #If the conditions are applied to a Wall only the viscous term is present
              #Therefore there is no need to calculate the derivatives
              
              #dP/dx and dP/dy from Navier-Stokes
              BCx=mu*LAPU
              BCy=mu*LAPV
              BCk=BCx*nx+BCy*ny
              
              #The gradient of the pressure is projected on 
              #the normal direction to the boundary
              del BCx,BCy
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X(X_C,Y_C,XBC[k],YBC[k],c,rcond)
              DY_P=Der_RBF_Y(X_C,Y_C,XBC[k],YBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T                           
              del DY_P,DX_P 
        else:
        #%% Dirichlet conditions
          #This is pretty straightforward the Dirichlet conditions are saved 
          #And matrix is simply calculated evaluating PHI over the constrained points
          BCk=BCD[k]
          MAT_CONk=PHI(X_C,Y_C,XBC[k],YBC[k],c,rcond)
          
        if k==0:#The vector is initialized
          BC=np.copy(BCk.reshape(-1))
          MAT_CON=np.copy(MAT_CONk)
          del MAT_CONk
        else:
            #The arrays are stacked
          BC=np.hstack((BC,BCk))
          MAT_CON=np.vstack((MAT_CON,MAT_CONk))
          del MAT_CONk
          
    return BC,MAT_CON

def Boundary_Conditions3D(rho,mu,XBC,YBC,ZBC,BCD,n,W_u,W_v,W_w,X_C_vel,Y_C_vel,Z_C_vel,cvel,X_C,Y_C,Z_C,c,rcond):
    """
   This function calculates the Boundary conditions and the condition matrix for the problem
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
                        
     :param XBC: tuple, It contains in its cells the one dimensional array of the point in X of every
                 boundary
                        
     :param YBC: tuple, It contains in its cells the one dimensional array of the point in Y of every
                 boundary
                 
     :param ZBC: tuple, It contains in its cells the one dimensional array of the point in Z of every
                 boundary
                        
     :param BCD: tuple, It contains only the Dirichlet conditions , in the cells of the tuple corresponding
                to a Neumann condition instead of an array with the conditions simply a string 'Neumann'
                has to be added. In order, to perform better if the Neumann condition is apply to a wall
                is preferable to plug 'Neumann Wall'
     
     :param n: tuple, contains the normal normalized vector evaluated in the corresponding point of XBC and YBC for every edge which has Neumann condition.
               If Dirichlet condition are present its cell can be fullfilled with everything. However is not possible to let it empty.
               The 2D matrix to insert in every cell must have shape 3XN where N is the lenght of the equivalent cell of XBC
               the rows are then nx,ny,nz at [XBC,YBC,ZBC]
               To remember the normal vector norm in any point has to be one. 
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
     
     :param W_w: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param Z_C_vel: array,Collocation point z of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure
     
     :param Z_C: array,Collocation point z of the pressure
     
     :param c: array,Shape parameter pressure
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return BC: array boundary condition  for every point
     :return MAT_CON: array, matrix of constraint
   """ 
   #%% Loop to create the boundary conditions in every edge
    for k in np.arange(0,len(XBC)):
        #if the variable BC is a string therefore the condition is Neumann or Neumann Wall
        if isinstance(BCD[k], str):
           #Create differentation matrices needed to construct the momentum equation
           LAP_vel=LAP_RBF3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
           
           #Evaluate the laplacian for all velocities
           LAPU=LAP_vel.dot(W_u)
           LAPV=LAP_vel.dot(W_v)
           LAPW=LAP_vel.dot(W_w)
           del LAP_vel
           
           #copy the normals to avoid damaging it
           nn=np.copy(n[k])
           #Extract the normal direction for x and y
           nx=nn[0,:]
           ny=nn[1,:]
           nz=nn[2,:]
           #%% Neumann conditions
           if BCD[k]=='Neumann':
              #Evaluate the X derivatives
              DX_vel=Der_RBF_X3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UX=DX_vel.dot(W_u)
              VX=DX_vel.dot(W_v)
              WX=DX_vel.dot(W_w)
              del DX_vel
              
              #Evaluate the Y derivatives
              DY_vel=Der_RBF_Y3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UY=DY_vel.dot(W_u)
              VY=DY_vel.dot(W_v)
              WY=DY_vel.dot(W_w)
              del DY_vel
              
              #Evaluate the Z derivatives
              DZ_vel=Der_RBF_Z3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UZ=DZ_vel.dot(W_u)
              VZ=DZ_vel.dot(W_v)
              WZ=DZ_vel.dot(W_w)
              del DZ_vel
              
              #Evaluate the velocities
              PHI_vel=PHI3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              U=PHI_vel.dot(W_u)
              V=PHI_vel.dot(W_v)
              W=PHI_vel.dot(W_w)
              del PHI_vel
              
              #dP/dx and dP/dy from Navier-Stokes
              BCx=-rho*(U*UX+V*UY+W*UZ)+mu*LAPU
              BCy=-rho*(U*VX+V*VY+W*VZ)+mu*LAPV
              BCz=-rho*(U*WX+V*WY+W*WZ)+mu*LAPW
              
              
              #The gradient of the pressure is projected on 
              #the normal direction to the boundary
              BCk=BCx*nx+BCy*ny+BCz*nz
              del BCx,BCy,BCz
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DY_P=Der_RBF_Y3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DZ_P=Der_RBF_Z3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T+(DZ_P.T*nz).T
              del DX_P,DY_P,DZ_P
              
           if BCD[k]=='Neumann Wall':
              #If the conditions are applied to a Wall only the viscous term is present
              #Therefore there is no need to calculate the derivatives
              
              #dP/dx and dP/dy from Navier-Stokes
              BCx=mu*LAPU
              BCy=mu*LAPV
              BCz=mu*LAPW
              BCk=BCx*nx+BCy*ny+BCz*nz
              del BCx,BCy,BCz
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DY_P=Der_RBF_Y3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DZ_P=Der_RBF_Z3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T+(DZ_P.T*nz).T
              del DX_P,DY_P,DZ_P
        else:
        #%% Dirichlet conditions
          #This is pretty straightforward the Dirichlet conditions are saved 
          #And matrix is simply calculated evaluating PHI over the constrined points
          BCk=BCD[k]
          MAT_CONk=PHI3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
          
        if k==0:#The vector is initialized
          BC=np.copy(BCk)
          MAT_CON=np.copy(MAT_CONk)
          del MAT_CONk
          
        else:
            #The arrays are stacked
          BC=np.hstack((BC,BCk))
          MAT_CON=np.vstack((MAT_CON,MAT_CONk))
          del MAT_CONk
          
    return BC,MAT_CON
def Boundary_Conditions3DRSI(rho,mu,XBC,YBC,ZBC,BCD,n,W_u,W_v,W_w,W_RSI,X_C_vel,Y_C_vel,Z_C_vel,cvel,X_C,Y_C,Z_C,c,rcond):
    """
   This function calculates the Boundary conditions and the condition matrix for the problem
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
                        
     :param XBC: tuple, It contains in its cells the one dimensional array of the point in X of every
                 boundary
                        
     :param YBC: tuple, It contains in its cells the one dimensional array of the point in Y of every
                 boundary
                 
     :param ZBC: tuple, It contains in its cells the one dimensional array of the point in Z of every
                 boundary
                        
     :param BCD: tuple, It contains only the Dirichlet conditions , in the cells of the tuple corresponding
                to a Neumann condition instead of an array with the conditions simply a string 'Neumann'
                has to be added. In order, to perform better if the Neumann condition is apply to a wall
                is preferable to plug 'Neumann Wall'
     
     :param n: tuple, contains the normal normalized vector evaluated in the corresponding point of XBC and YBC for every edge which has Neumann condition.
               If Dirichlet condition are present its cell can be fullfilled with everything. However is not possible to let it empty.
               The 2D matrix to insert in every cell must have shape 3XN where N is the lenght of the equivalent cell of XBC
               the rows are then nx,ny,nz at [XBC,YBC,ZBC]
               To remember the normal vector norm in any point has to be one. 
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
     
     :param W_w: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param Z_C_vel: array,Collocation point z of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure
     
     :param Z_C: array,Collocation point z of the pressure
     
     :param c: array,Shape parameter pressure
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return BC: array boundary condition  for every point
     :return MAT_CON: array, matrix of constraint
   """ 
   #%% Loop to create the boundary conditions in every edge
    for k in np.arange(0,len(XBC)):
        #if the variable BC is a string therefore the condition is Neumann or Neumann Wall
        if isinstance(BCD[k], str):
           #Create differentation matrices needed to construct the momentum equation
           LAP_vel=LAP_RBF3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
           
           #Evaluate the laplacian for all velocities
           LAPU=LAP_vel.dot(W_u)
           LAPV=LAP_vel.dot(W_v)
           LAPW=LAP_vel.dot(W_w)
           del LAP_vel
           
           #copy the normals to avoid damaging it
           nn=np.copy(n[k])
           #Extract the normal direction for x and y
           nx=nn[0,:]
           ny=nn[1,:]
           nz=nn[2,:]
           #%% Neumann conditions
           if BCD[k]=='Neumann':
              #Evaluate the X derivatives
              DX_vel=Der_RBF_X3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UX=DX_vel.dot(W_u)
              VX=DX_vel.dot(W_v)
              WX=DX_vel.dot(W_w)
              del DX_vel
              
              #Evaluate the Y derivatives
              DY_vel=Der_RBF_Y3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UY=DY_vel.dot(W_u)
              VY=DY_vel.dot(W_v)
              WY=DY_vel.dot(W_w)
              del DY_vel
              
              #Evaluate the Z derivatives
              DZ_vel=Der_RBF_Z3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UZ=DZ_vel.dot(W_u)
              VZ=DZ_vel.dot(W_v)
              WZ=DZ_vel.dot(W_w)
              del DZ_vel
              
              #Evaluate the velocities
              PHI_vel=PHI3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              U=PHI_vel.dot(W_u)
              V=PHI_vel.dot(W_v)
              W=PHI_vel.dot(W_w)
              del PHI_vel
              
              #dP/dx and dP/dy from Navier-Stokes
              BCx=-rho*(U*UX+V*UY+W*UZ)+mu*LAPU
              BCy=-rho*(U*VX+V*VY+W*VZ)+mu*LAPV
              BCz=-rho*(U*WX+V*WY+W*WZ)+mu*LAPW
              
              
              #The gradient of the pressure is projected on 
              #the normal direction to the boundary
              BCk=BCx*nx+BCy*ny+BCz*nz
              del BCx,BCy,BCz
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DY_P=Der_RBF_Y3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DZ_P=Der_RBF_Z3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T+(DZ_P.T*nz).T
              del DX_P,DY_P,DZ_P
              
           if BCD[k]=='Neumann Wall':
              #If the conditions are applied to a Wall only the viscous term is present
              #Therefore there is no need to calculate the derivatives
              #dP/dx and dP/dy from Navier-Stokes
              BCx=mu*LAPU
              BCy=mu*LAPV
              BCz=mu*LAPW
              BCk=BCx*nx+BCy*ny+BCz*nz
              del BCx,BCy,BCz
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DY_P=Der_RBF_Y3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DZ_P=Der_RBF_Z3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T+(DZ_P.T*nz).T
              del DX_P,DY_P,DZ_P
        else:
        #%% Dirichlet conditions
          #This is pretty straightforward the Dirichlet conditions are saved 
          #And matrix is simply calculated evaluating PHI over the constrined points
          BCk=BCD[k]+rho*PHI3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)@W_RSI
          MAT_CONk=PHI3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
          
        if k==0:#The vector is initialized
          BC=np.copy(BCk)
          MAT_CON=np.copy(MAT_CONk)
          del MAT_CONk
          
        else:
            #The arrays are stacked
          BC=np.hstack((BC,BCk))
          MAT_CON=np.vstack((MAT_CON,MAT_CONk))
          del MAT_CONk
          
    return BC,MAT_CON
def Boundary_Conditions3DRSA(rho,mu,XBC,YBC,ZBC,BCD,n,W_u,W_v,W_w,W_RSXX,W_RSXY,W_RSXZ,W_RSYY,W_RSYZ,W_RSZZ,X_C_vel,Y_C_vel,Z_C_vel,cvel,X_C,Y_C,Z_C,c,rcond):
    """
   This function calculates the Boundary conditions and the condition matrix for the problem
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param rho: float, density in coherent unit measurement
                        
     :param mu: float,dynamic viscosity in coherent unit measurement
                        
     :param XBC: tuple, It contains in its cells the one dimensional array of the point in X of every
                 boundary
                        
     :param YBC: tuple, It contains in its cells the one dimensional array of the point in Y of every
                 boundary
                 
     :param ZBC: tuple, It contains in its cells the one dimensional array of the point in Z of every
                 boundary
                        
     :param BCD: tuple, It contains only the Dirichlet conditions , in the cells of the tuple corresponding
                to a Neumann condition instead of an array with the conditions simply a string 'Neumann'
                has to be added. In order, to perform better if the Neumann condition is apply to a wall
                is preferable to plug 'Neumann Wall'
     
     :param n: tuple, contains the normal normalized vector evaluated in the corresponding point of XBC and YBC for every edge which has Neumann condition.
               If Dirichlet condition are present its cell can be fullfilled with everything. However is not possible to let it empty.
               The 2D matrix to insert in every cell must have shape 3XN where N is the lenght of the equivalent cell of XBC
               the rows are then nx,ny,nz at [XBC,YBC,ZBC]
               To remember the normal vector norm in any point has to be one. 
          
     :param W_u: array,array of u velocity weights
                 
     :param W_v: array,array of v velocity weights
     
     :param W_w: array,array of v velocity weights
                 
     :param X_C_vel: array,Collocation point x of the velocity interpolation
     
     :param Y_C_vel: array,Collocation point y of the velocity interpolation
     
     :param Z_C_vel: array,Collocation point z of the velocity interpolation
     
     :param cvel: array,Shape parameters for the velocity
     
     :param X_C: array,Collocation point x of the pressure
     
     :param Y_C: array,Collocation point y of the pressure
     
     :param Z_C: array,Collocation point z of the pressure
     
     :param c: array,Shape parameter pressure
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return BC: array boundary condition  for every point
     :return MAT_CON: array, matrix of constraint
   """ 
   #%% Loop to create the boundary conditions in every edge
    for k in np.arange(0,len(XBC)):
        #if the variable BC is a string therefore the condition is Neumann or Neumann Wall
        if isinstance(BCD[k], str):
           #Create differentation matrices needed to construct the momentum equation
           LAP_vel=LAP_RBF3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
           
           #Evaluate the laplacian for all velocities
           LAPU=LAP_vel.dot(W_u)
           LAPV=LAP_vel.dot(W_v)
           LAPW=LAP_vel.dot(W_w)
           del LAP_vel
           
           #copy the normals to avoid damaging it
           nn=np.copy(n[k])
           #Extract the normal direction for x and y
           nx=nn[0,:]
           ny=nn[1,:]
           nz=nn[2,:]
           #%% Neumann conditions
           if BCD[k]=='Neumann':
              #Evaluate the X derivatives
              DX_vel=Der_RBF_X3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UX=DX_vel.dot(W_u)
              VX=DX_vel.dot(W_v)
              WX=DX_vel.dot(W_w)
              RSXXX=DX_vel.dot(W_RSXX)
              RSXYX=DX_vel.dot(W_RSXY)
              RSXZX=DX_vel.dot(W_RSXZ)
              del DX_vel
              
              #Evaluate the Y derivatives
              DY_vel=Der_RBF_Y3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UY=DY_vel.dot(W_u)
              VY=DY_vel.dot(W_v)
              WY=DY_vel.dot(W_w)
              RSYYY=DY_vel.dot(W_RSYY)
              RSXYY=DY_vel.dot(W_RSXY)
              RSYZY=DY_vel.dot(W_RSYZ)
              del DY_vel
              
              #Evaluate the Z derivatives
              DZ_vel=Der_RBF_Z3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              UZ=DZ_vel.dot(W_u)
              VZ=DZ_vel.dot(W_v)
              WZ=DZ_vel.dot(W_w)
              RSXZZ=DZ_vel.dot(W_RSXZ)
              RSYZZ=DZ_vel.dot(W_RSYZ)
              RSZZZ=DZ_vel.dot(W_RSZZ)
              del DZ_vel
              
              #Evaluate the velocities
              PHI_vel=PHI3D(X_C_vel,Y_C_vel,Z_C_vel,XBC[k],YBC[k],ZBC[k],cvel,rcond)
              U=PHI_vel.dot(W_u)
              V=PHI_vel.dot(W_v)
              W=PHI_vel.dot(W_w)
              del PHI_vel
              
              #dP/dx and dP/dy from Navier-Stokes
              BCx=-rho*(U*UX+V*UY+W*UZ)+mu*LAPU-rho*(RSXXX+RSXYY+RSXZZ)
              BCy=-rho*(U*VX+V*VY+W*VZ)+mu*LAPV-rho*(RSXYX+RSYYY+RSYZZ)
              BCz=-rho*(U*WX+V*WY+W*WZ)+mu*LAPW-rho*(RSXZX+RSYZY+RSZZZ)
              
              
              #The gradient of the pressure is projected on 
              #the normal direction to the boundary
              BCk=BCx*nx+BCy*ny+BCz*nz
              del BCx,BCy,BCz
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DY_P=Der_RBF_Y3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DZ_P=Der_RBF_Z3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T+(DZ_P.T*nz).T
              del DX_P,DY_P,DZ_P
              
           if BCD[k]=='Neumann Wall':
              #If the conditions are applied to a Wall only the viscous term is present
              #Therefore there is no need to calculate the derivatives
              #dP/dx and dP/dy from Navier-Stokes
              BCx=mu*LAPU
              BCy=mu*LAPV
              BCz=mu*LAPW
              BCk=BCx*nx+BCy*ny+BCz*nz
              del BCx,BCy,BCz
              
              #Projection of the pressure derivatives into the n direction
              DX_P=Der_RBF_X3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DY_P=Der_RBF_Y3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              DZ_P=Der_RBF_Z3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
              MAT_CONk=(DX_P.T*nx).T+(DY_P.T*ny).T+(DZ_P.T*nz).T
              del DX_P,DY_P,DZ_P
        else:
        #%% Dirichlet conditions
          #This is pretty straightforward the Dirichlet conditions are saved 
          #And matrix is simply calculated evaluating PHI over the constrined points
          BCk=BCD[k]
          MAT_CONk=PHI3D(X_C,Y_C,Z_C,XBC[k],YBC[k],ZBC[k],c,rcond)
          
        if k==0:#The vector is initialized
          BC=np.copy(BCk)
          MAT_CON=np.copy(MAT_CONk)
          del MAT_CONk
          
        else:
            #The arrays are stacked
          BC=np.hstack((BC,BCk))
          MAT_CON=np.vstack((MAT_CON,MAT_CONk))
          del MAT_CONk
          
    return BC,MAT_CON