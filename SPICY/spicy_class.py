# -*- coding: utf-8 -*-
"""
Latest update on Thu Jan 12 17:56:06 2023

@author: mendez, ratz, sperotto
"""

# Test
from SPICY.Matrix import Der_RBF_X,Der_RBF_Y,PHI3D,Der_RBF_X3D,Der_RBF_Y3D,Der_RBF_Z3D
from SPICY.Fit import Inter_2D_C,Inter_3D_C,Inter_3D_RSI
from SPICY.Extrapolate import Fit_vel2D,Fit_RBF,Fit_RBF3D,Fit_vel3D
from SPICY.Cluster import Clusteringmethod,Clusteringmethod3D
from SPICY.BC import Boundary_Conditions,Boundary_Conditions3D,Boundary_Conditions3DRSA,Boundary_Conditions3DRSI
from SPICY.MeshPoiss import Poisson_solver,Poisson_solver3D,Poisson_solver3DRSA
import numpy as np
from sklearn.neighbors import NearestNeighbors

def roundDown(x): 
    # Round near numbers to avoid that 15 becomes 14.99999999994
    xtemp=np.copy(x)
    xtemp[x==0]=1
    exponent = np.ceil(np.log10(np.abs(xtemp))) 
    mantissa = x/(10**exponent) #get full precision mantissa
    # change floor here to ceil or round to round up or to zero
    mantissa = mantissa.round(decimals=15)
    xnew=mantissa * 10**exponent
    xnew[x==0]=0
    return xnew

#Calculate the scaler
def scaling(X,scaler):
    Xnew=[X[0]/scaler]
    for k in np.arange(1,len(X)):
        Xnew.append(X[k]/scaler)
    return Xnew  

    
class spicy:
    def __init__(self,model,Velocities,grid_point,ST=None):
        """
        Definition of the class spicy
             
        # The input parameters are 
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param model: string
                   Currently, SPICY supports 4 possible models:
                       1. 'scalar', to regress a scalar quantity.
                       2. 'laminar', to regress the velocity field without turbulence modeling.
                       3.  'RANSI', to regress a velocity field with a RANS model assuming isotropic Reynolds stresses (hence mean(u'**2)) is the only extra expected quantity.
                            This must be provided as the fourth entry of 'velocities', which becomes
                       4.  'RANSA', to regress a velocity field with a RANS model without assuming isotropic Reynolds stresses.
                            this becomes [uu, vv, uv] in 2D and [uu, vv, ww, uv, uw, vw] in 3D.
        :param Velocities: list
                    Is a list of arrays containing [u,v,w] if the flow is 3D otherwise just [u,v]
                                        
        :param grid_point: list
                    Is a list of arrays containing the grid point [XG,YG,ZG] in 3D and [XG,YG] in 2D.   
                
        :param ST: list
                    Is a list of arrays collecting Reynolds stresses. This is empty if the model is 'scalar' or 'laminar'.
                    If the model is RANSI, it contains [uu']. 
                    If the model is RANSA, it contains [uu, vv, uv] in 2D and [uu, vv, ww, uv, uw, vw] in 3D.                                                   
                            
        ----------------------------------------------------------------------------------------------------------------
        """
        self.model=model
        
        if len(grid_point)==2:
            self.type='2D'
            self.u=Velocities[0]
            self.v=Velocities[1]
            self.XG=grid_point[0]
            self.YG=grid_point[1]
            
        if len(grid_point)==3:
            self.type='3D'
            self.u=Velocities[0]
            self.v=Velocities[1]
            self.w=Velocities[2]
            self.XG=grid_point[0]
            self.YG=grid_point[1]
            self.ZG=grid_point[2]            
            
            if model=='RANSI':
            
             self.RSI=Velocities[3]
             self.Constr_RS=np.array([])
             self.XCONRS=np.array([])
             self.YCONRS=np.array([])
             self.ZCONRS=np.array([])
             
            if model=='RANSA':
            
             self.RSX=ST[0]
             self.RSY=ST[1]
             self.RSZ=ST[2]
             self.RSXY=ST[3]
             self.RSXZ=ST[4]
             self.RSYZ=ST[5]
             self.Constr_RS=np.array([])
             self.XCONRS=np.array([])
             self.YCONRS=np.array([])
             self.ZCONRS=np.array([])
        return
    
    


#%% 1. Clustering (this does not depend on the model, but only on the dimension).
def clustering:
    
    
    return


#%% 2. Constraints (this depends on everything)
def constraints:
    
    return

#%% 3. Assembly A, B, C(this depends on everything)

def Assembly:
    
    return








    
    
    
    
    
    
    
#%% The lines below are coming from Pietro. To be restructured in the new architecture

    
    
    
    def RS_constraint_definition(self,constraint):
        
        if self.type=='3D':
         XC=constraint[0]
         YC=constraint[1]
         ZC=constraint[2]
         CON=constraint[3]
         i=False
         for k in np.arange(len(XC)):
             if i:
                 self.XCONRS=np.hstack((self.XCONRS,XC[k]))
                 self.YCONRS=np.hstack((self.YCONRS,YC[k]))
                 self.ZCONRS=np.hstack((self.ZCONRS,ZC[k]))
                 self.Constr_RS=np.hstack((self.Constr_RS,CON[k]))
             else:
                 self.XCONRS=XC[k]
                 self.YCONRS=YC[k]
                 self.ZCONRS=ZC[k]
                 self.Constr_RS=CON[k]
                 i=True
             
           
    def velocities_constraint_definition(self,constraint):
      """
        save the constraint in a way such that they can work with the functions
             
        
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param constraint: tuple,
                    In the first two or three cell depending on the dimensionality contains a list containing again the arrays of every
                    boundary or constrained points. The last cell is a list containing a list of 2 or 3 list depending on the dimensionality contains a list containing again the arrays of every
                    boundary or constrained points. Sincerely don't know how to explain this check the test case'
                    
        
        ----------------------------------------------------------------------------------------------------------------
      """
      #If the flow is two diminsional
      if self.type=='2D':
         
         #Inizialiting the value extracting the thing and assigning void arrays to the various values
         XC=constraint[0]
         YC=constraint[1]
         CON=constraint[2]
         CONU=CON[0]
         CONV=CON[1]
         self.XDIV=np.array([])
         self.YDIV=np.array([])
         self.XCON=np.array([])
         self.YCON=np.array([])
         self.Constr_u=np.array([])
         self.Constr_v=np.array([])
         
         #Assigning False to this values, they are dummy to control if XDIV or 
         #XCON has been assigned
         i=False
         ii=False
         
         #Loop to assign correctly every part
         for k in np.arange(len(XC)):
             
             if k==0:
             #Inizializing the arrays
                 #Assigning the values to XCON1 this arrays contains all the 
                 #constraints indipendently id they are classic contraint or
                 #pure divergence-free
                 self.XCON1=XC[k]
                 self.YCON1=YC[k]
                 
                 #If the relative constraint contains a string (only_div)
                 #The values are assigned as XDIV so pure divergence-free
                 if isinstance(CONU[k], str) or isinstance(CONV[k], str):
                     self.XDIV=XC[k]
                     self.YDIV=YC[k]
                     i=True#Set to true means that XDIV is not empty
                     
                 else:
                  self.XCON=XC[k]
                  self.YCON=YC[k]  
                  self.Constr_u=CONU[k]
                  self.Constr_v=CONV[k]
                  ii=True#Set to true means that XCON is not empty
             else:
             #Stacking the arrays
                 #Assigning the values to XCON1 this arrays contains all the 
                 #constraints indipendently id they are classic contraint or
                 #pure divergence-free            
                 self.XCON1=np.hstack((self.XCON1,XC[k]))
                 self.YCON1=np.hstack((self.YCON1,YC[k])) 
                 
                 #If the relative constraint contains a string (only_div)
                 #The values are assigned as XDIV so pure divergence-free
                 if isinstance(CONU[k], str) or isinstance(CONV[k], str):
                  #Another if is added to see if XDIV has been inizialize 
                  #already
                  if i:
                   self.XDIV=np.hstack((self.XDIV,XC[k]))
                   self.YDIV=np.hstack((self.YDIV,YC[k]))
                  else:
                   self.XDIV=XC[k]
                   self.YDIV=YC[k]   
                   i=True#Set to true means that XDIV is not empty
                   
                 else:
                  #Another if is added to see if XCON has been inizialize 
                  #already                   
                   if ii:
                     self.XCON=np.hstack((self.XCON,XC[k]))
                     self.YCON=np.hstack((self.YCON,YC[k]))
                     self.Constr_u=np.hstack((self.Constr_u,CONU[k]))
                     self.Constr_v=np.hstack((self.Constr_v,CONV[k]))
                   else:
                    self.XCON=XC[k]
                    self.YCON=YC[k]  
                    self.Constr_u=CONU[k]
                    self.Constr_v=CONV[k]
                    ii=True#Set to true means that XCON is not empty
                    
         #If the XCON has been fulfilled and they are repeated if the condition
         #Is the same one is just delete otherwise the code give an error for 
         #incopatible constraint
         if ii:
             BBB=np.unique(np.column_stack((self.XCON,self.YCON,self.Constr_u,self.Constr_v)),axis=0)
             self.XCON=BBB[:,0]
             self.YCON=BBB[:,1]
             self.Constr_u=BBB[:,2]
             self.Constr_v=BBB[:,3]
             BBB=np.unique(np.column_stack((roundDown(self.XCON),roundDown(self.YCON))),axis=0)
             if len(BBB[:,0])!=len(self.XCON):
                raise ValueError('Some constraints are incompatible')
                
         #Repeated conditions are the deleted (in pure divergence the condition)
         #cannot be incopatible
         BBB=np.unique(np.column_stack((roundDown(self.XCON1),roundDown(self.YCON1))),axis=0)
         self.XCON1=BBB[:,0]
         self.YCON1=BBB[:,1]
         
         # A similar condition is added to check XDIV
         if i:
          BBB=np.unique(np.column_stack((roundDown(self.XDIV),roundDown(self.YDIV))),axis=0)
          self.XDIV=BBB[:,0]
          self.YDIV=BBB[:,1]
      #If the flow is three dimensional
      if self.type=='3D':
          
         #Inizialiting the value extracting the thing and assigning void arrays to the various values 
         XC=constraint[0]
         YC=constraint[1]
         ZC=constraint[2]
         CON=constraint[3]
         CONU=CON[0]
         CONV=CON[1]
         CONW=CON[2]
         self.XDIV=np.array([])
         self.YDIV=np.array([])
         self.ZDIV=np.array([])
         self.XCON=np.array([])
         self.YCON=np.array([])
         self.ZCON=np.array([])
         self.Constr_u=np.array([])
         self.Constr_v=np.array([])
         self.Constr_w=np.array([])
         
         #Assigning False to this values, they are dummy to control if XDIV or 
         #XCON has been assigned
         i=False
         ii=False
         #Loop to assign correctly every part
         for k in np.arange(len(XC)):
          if k==0:
             #Inizializing the arrays
                 #Assigning the values to XCON1 this arrays contains all the 
                 #constraints indipendently id they are classic contraint or
                 #pure divergence-free
                 self.XCON1=XC[k]
                 self.YCON1=YC[k]
                 self.ZCON1=ZC[k]
                 #If the relative constraint contains a string (only_div)
                 #The values are assigned as XDIV so pure divergence-free                 
                 if isinstance(CONU[k], str) or isinstance(CONV[k], str) or isinstance(CONW[k], str):
                     self.XDIV=XC[k]
                     self.YDIV=YC[k]
                     self.ZDIV=ZC[k]
                     i=True#Set to true means that XDIV is not empty
                     
                 else: 
                  self.XCON=XC[k]
                  self.YCON=YC[k] 
                  self.ZCON=ZC[k]
                  self.Constr_u=CONU[k]
                  self.Constr_v=CONV[k]
                  self.Constr_w=CONW[k]
                  ii=True#Set to true means that XCON is not empty
          else:
          #Stacking the arrays
                 #Assigning the values to XCON1 this arrays contains all the 
                 #constraints indipendently id they are classic contraint or
                 #pure divergence-free 
                 self.XCON1=np.hstack((self.XCON1,XC[k]))
                 self.YCON1=np.hstack((self.YCON1,YC[k])) 
                 self.ZCON1=np.hstack((self.ZCON1,ZC[k]))
                 #If the relative constraint contains a string (only_div)
                 #The values are assigned as XDIV so pure divergence-free
                 if isinstance(CONU[k], str) or isinstance(CONV[k], str) or isinstance(CONW[k], str):
                #Another if is added to see if XDIV has been inizialize 
                #already
                  if i:
                   self.XDIV=np.hstack((self.XDIV,XC[k]))
                   self.YDIV=np.hstack((self.YDIV,YC[k]))
                   self.ZDIV=np.hstack((self.ZDIV,ZC[k]))
                  else:
                   self.XDIV=XC[k]
                   self.YDIV=YC[k]   
                   self.ZDIV=ZC[k]  
                   i=True#Set to true means that XDIV is not empty
                 else:
                #Another if is added to see if XCON has been inizialize 
                #already
                   if ii:
                     
                     self.XCON=np.hstack((self.XCON,XC[k]))
                     self.YCON=np.hstack((self.YCON,YC[k]))
                     self.ZCON=np.hstack((self.ZCON,ZC[k]))
                     self.Constr_u=np.hstack((self.Constr_u,CONU[k]))
                     self.Constr_v=np.hstack((self.Constr_v,CONV[k]))
                     self.Constr_w=np.hstack((self.Constr_w,CONW[k]))
                   else:
                    self.XCON=XC[k]
                    self.YCON=YC[k]  
                    self.ZCON=ZC[k] 
                    self.Constr_u=CONU[k]
                    self.Constr_v=CONV[k]
                    self.Constr_w=CONW[k]
                    ii=True#Set to true means that XCON is not empty
                    
        #If the XCON has been fulfilled and they are repeated if the condition
        #Is the same one is just delete otherwise the code give an error for 
        #incompatible constraint           
         if ii:
             BBB=np.unique(np.column_stack((roundDown(self.XCON),roundDown(self.YCON),roundDown(self.ZCON),roundDown(self.Constr_u),roundDown(self.Constr_v),roundDown(self.Constr_w))),axis=0)
             self.XCON=BBB[:,0]
             self.YCON=BBB[:,1]
             self.ZCON=BBB[:,2]
             self.Constr_u=BBB[:,3]
             self.Constr_v=BBB[:,4]
             self.Constr_w=BBB[:,5]
             BBB=np.unique(np.column_stack((roundDown(self.XCON),roundDown(self.YCON),roundDown(self.ZCON))),axis=0)
             if len(BBB[:,0])!=len(self.XCON):
                raise ValueError('Some constraints are incompatible')
                
         #Repeated conditions are the deleted (in pure divergence the condition)
         #cannot be incopatible
         BBB=np.unique(np.column_stack((roundDown(self.XCON1),roundDown(self.YCON1),roundDown(self.ZCON1))),axis=0)
         self.XCON1=BBB[:,0]
         self.YCON1=BBB[:,1]
         self.ZCON1=BBB[:,2]
         # A similar condition is added to check XDIV
         if i:
             BBB=np.unique(np.column_stack((roundDown(self.XDIV),roundDown(self.YDIV),roundDown(self.ZDIV))),axis=0)
             self.XDIV=BBB[:,0]
             self.YDIV=BBB[:,1]
             self.ZDIV=BBB[:,2]
      
      return
      
    def clustering_velocities(self,N,cap,mincluster=[False],el=np.exp(-0.5**2/2),collocation_augmentation=0):
        """
          calculate the collocation points by using the clustering method           
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param N: list,
                    list of int , it contains the number of point per Gaussian 
          :param cap: float,
                    maximum shape factors required.
          :param mincluster: list (optional),
                    list of bool if set to true means that if N=4 to every Gaussian 
                    where N<4 are assigned with the maximum c of the level.
          :param el: float (optional),
                     the values that the Gaussian must reach in the nearest collocation point
                     typically beetwen 0.7-0.9.
          ----------------------------------------------------------------------------------------------------------------
        """
        
        #save the elongation in the class
        self.el=el
        self.N=N
        self.mincluster=mincluster
        #save the cap in the class
        self.cap=cap
        #If two dimensional
        if self.type=='2D':
            
         #Recall clustering method two build the collocation point
         self.X_C,self.Y_C,self.c=Clusteringmethod(self.XG,self.YG,N,self.el,cap,mincluster)
         
         #Inizializing the distance of the constraint point
         distances=np.zeros(len(self.XCON1))
         
         #seeking for the nearest neighboor of the constrained point
         #in the constrained points and in the grid points
         for k in np.arange(len(self.XCON1)):
             distances1=np.sqrt((self.X_C-self.XCON1[k])**2+(self.Y_C-self.YCON1[k])**2)
             distances2=np.sqrt((np.delete(self.XCON1,k)-self.XCON1[k])**2+(np.delete(self.YCON1,k)-self.YCON1[k])**2)
             distances1=distances1[distances1>np.sqrt(-np.log(self.el))/self.cap]
             distances2=distances2[distances2>np.sqrt(-np.log(self.el))/self.cap]
             distances[k]=min(np.min(distances1),np.min(distances2))
             
         #define the shape parameter for the constrained point
         cC=np.sqrt(-np.log(self.el))/(distances)
         cC[cC>self.cap]=self.cap
         
         #Merging the result in one array
         self.cvel=np.hstack((self.c,cC))
         self.X_C_vel=np.hstack((self.X_C,self.XCON1))
         self.Y_C_vel=np.hstack((self.Y_C,self.YCON1))
        
        #If three dimensional
        if self.type=='3D':
         if N[0]==1:
             if len(N)>1:
              N=N[1::]
              self.X_C,self.Y_C,self.Z_C,self.c=Clusteringmethod3D(self.XG,self.YG,self.ZG,N,self.el,cap,mincluster)
              Centers=np.column_stack((self.XG,self.YG,self.ZG))
              nbrs = NearestNeighbors(n_neighbors=2,algorithm='ball_tree',n_jobs=8).fit(Centers)#Calculate the nearest neighbor for the centers
              distances, indices = nbrs.kneighbors(Centers)
              sigma1=distances[:,1]
              cplus=np.sqrt(-np.log(self.el))/(sigma1)
              cplus[cplus>cap]=cap
              self.X_C=np.hstack((self.X_C,self.XG))
              self.Y_C=np.hstack((self.Y_C,self.YG))
              self.Z_C=np.hstack((self.Z_C,self.ZG))
              self.c=np.hstack((self.c,cplus))
             else:  
              Centers=np.column_stack((self.XG,self.YG,self.ZG))   
              nbrs = NearestNeighbors(n_neighbors=2,algorithm='ball_tree',n_jobs=8).fit(Centers)#Calculate the nearest neighbor for the centers
              distances, indices = nbrs.kneighbors(Centers)
              sigma1=distances[:,1]
              self.c=np.sqrt(-np.log(self.el))/(sigma1)
              self.c[self.c>cap]=cap
              self.X_C=self.XG
              self.Y_C=self.YG
              self.Z_C=self.ZG
         else:     
         #Recall clustering method two build the collocation point 
          self.X_C,self.Y_C,self.Z_C,self.c=Clusteringmethod3D(self.XG,self.YG,self.ZG,N,self.el,cap,mincluster)
         
         #Inizializing the distance of the constraint point
         distances=np.zeros(len(self.XCON1))
         
         #seeking for the nearest neighboor of the constrained point
         #in the constrained points and in the grid points
         for k in np.arange(len(self.XCON1)):
             distances1=np.sqrt((self.X_C-self.XCON1[k])**2+(self.Y_C-self.YCON1[k])**2+(self.Z_C-self.ZCON1[k])**2)
             distances2=np.sqrt((np.delete(self.XCON1,k)-self.XCON1[k])**2+(np.delete(self.YCON1,k)-self.YCON1[k])**2+(np.delete(self.ZCON1,k)-self.ZCON1[k])**2)
             distances1=distances1[distances1>np.sqrt(-np.log(self.el))/self.cap]
             distances2=distances2[distances2>np.sqrt(-np.log(self.el))/self.cap]
             distances[k]=max(np.min(distances1),np.min(distances2))
             
         #define the shape parameter for the constrained point    
         cC=np.sqrt(-np.log(self.el))/(distances)
         cC[cC>self.cap]=self.cap
         
         #Merging the result in one array
         self.cvel=np.hstack((self.c,cC))
         self.X_C_vel=np.hstack((self.X_C,self.XCON1))
         self.Y_C_vel=np.hstack((self.Y_C,self.YCON1))
         self.Z_C_vel=np.hstack((self.Z_C,self.ZCON1))
        return
    
    def approximation_velocities(self,DIV=1,rcond=1e-13,method='fullcho'):
        self.rcond_vel=rcond
        """
          compute the approximations of the velocities
               
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
        :param DIV: float (optional) 
                    0 if no Divergence penalty 1 if there is
                    
        :param rcond: float (optional)
                    inverse of the maximum conditioning acceptable
        
        :param method: string (optional) 
                    default is 'fullcho' , other option are 'mixed' and 'fullpinv'. 'fullcho' uses only Cholensky
                    'fullpinv' uses only pseudo inverse and finally 'mixed' uses Cholensky for the variance matrix
                    and a pseudoinverse for the constraint part. Finally, 'solve' use np.linalg.solve on the complete matrix.
                      
          
          ----------------------------------------------------------------------------------------------------------------
        """
        #The right function is recalled in base of simensionality
        
        if self.type=='2D':
         
         self.w_u,self.w_v=Inter_2D_C(self.u,self.v,self.XG,self.YG,self.X_C_vel,self.Y_C_vel,self.cvel,self.XCON,self.YCON,self.Constr_u,self.Constr_v,self.XDIV,self.YDIV,self.rcond_vel,DIV,method)
        if self.type=='3D':
         self.w_u,self.w_v,self.w_w=Inter_3D_C(self.u,self.v,self.w,self.XG,self.YG,self.ZG,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.XCON,self.YCON,self.ZCON,self.Constr_u,self.Constr_v,self.Constr_w,self.XDIV,self.YDIV,self.ZDIV,self.rcond_vel,DIV,method)
         if self.model=='RANSI':
             #If the model is RANSI then the reynolds stresses must be calculated
             self.W_RSI=Inter_3D_RSI(self.RSI,self.XG,self.YG,self.ZG,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.XCONRS,self.YCONRS,self.ZCONRS,self.Constr_RS,self.rcond_vel,method)
         if self.model=='RANSA':
             #If the model is RANSI then the reynolds stresses must be calculated
             W_RSI=Inter_3D_RSI(np.vstack((self.RSX,self.RSXY,self.RSXZ,self.RSY,self.RSYZ,self.RSZ)).T,self.XG,self.YG,self.ZG,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.XCONRS,self.YCONRS,self.ZCONRS,self.Constr_RS,self.rcond_vel,method)
             self.W_RSXX=W_RSI[:,0]
             self.W_RSXY=W_RSI[:,1]
             self.W_RSXZ=W_RSI[:,2]
             self.W_RSYY=W_RSI[:,3]
             self.W_RSYZ=W_RSI[:,4]
             self.W_RSZZ=W_RSI[:,5]
             del W_RSI
        return
    
    def pressure_boundary_conditions(self,rho,mu,boundary_condition,normal_direction):
        """
          save the boundary conditions
               
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param rho: float, density in coherent unit measurement
                        
          :param mu: float,dynamic viscosity in coherent unit measurement
                        
                        
          :param boundary_condition: tuple, Contains the position of the BC and the BC for example in 2D the first cell contains a list of array any cell of this last list is referred to
                                            one boundary in the second position the Y position and in the last one the boundary condition. These are arrays of the same length of the position
                                            or simply 'NEUMANN' for Neumann conditions or 'NEUMANN WALL' if applied on a wall
     
          :param normal_direction: tuple, contains the normal normalized vector evaluated in the corresponding point of XBC and YBC for every edge which has Neumann condition.
               If Dirichlet condition are present its cell can be fullfilled with everything. However is not possible to let it empty.
               The 2D matrix to insert in every cell must have shape 3XN where N is the lenght of the equivalent cell of XBC
               the rows are then nx,ny,nz at [XBC,YBC,ZBC]. The same is true for 2D case obviously adapting.
               To remember the normal vector norm in any point has to be one. .
                      
          
          ----------------------------------------------------------------------------------------------------------------
        """
        #Unpack 2D data
        if self.type=='2D':
            self.XBC=boundary_condition[0]
            self.YBC=boundary_condition[1]
            self.BCD=boundary_condition[2]
            self.rho=rho
            self.mu=mu
            self.n=normal_direction
            
        #Unpack 3D data    
        if self.type=='3D':
            self.XBC=boundary_condition[0]
            self.YBC=boundary_condition[1]
            self.ZBC=boundary_condition[2]
            self.BCD=boundary_condition[3]
            self.n=normal_direction
            
            #Loop to che for repetitions
            for k in np.arange(len(self.XBC)):
                BBB , index=np.unique(np.column_stack((roundDown(self.XBC[k]),roundDown(self.YBC[k]),roundDown(self.ZBC[k]))),axis=0,return_index=True)
                if type(self.BCD[k])!=str:
                  BCB=self.BCD[k]
                  self.BCD[k]=BCB[index]
                else:
                  nn=self.n[k]
                  self.n[k]=nn[:,index]
                  
                self.XBC[k]=BBB[:,0]
                self.YBC[k]=BBB[:,1]
                self.ZBC[k]=BBB[:,2]
            self.rho=rho
            self.mu=mu
        return

    def clustering_pressure(self,capp=0):
        """
          Create the collocation point for the pressure right now is assumed
          that the collocation point from the clustering are the same only the boundary point can change
        """
        #If 2D
        if capp==0:
            capp=self.cap
        if self.type=='2D':
            
         #Unpacking and eliminating double
         XBCC=np.hstack(self.XBC)
         YBCC=np.hstack(self.YBC)
         BBB=np.unique(np.column_stack((roundDown(XBCC),roundDown(YBCC))),axis=0)
         XBCC=BBB[:,0]
         YBCC=BBB[:,1]
         
         #Inizializing the distance of the constraint point
         distances=np.zeros(len(XBCC))
         
         #seeking for the nearest neighboor of the constrained point
         #in the constrained points and in the grid points
         for k in np.arange(len(XBCC)):
             distances1=np.sqrt((self.X_C-XBCC[k])**2+(self.Y_C-YBCC[k])**2)
             distances2=np.sqrt((np.delete(XBCC,k)-XBCC[k])**2+(np.delete(YBCC,k)-YBCC[k])**2)
             distances1=distances1[distances1>np.sqrt(-np.log(self.el))/self.cap]
             distances2=distances2[distances2>np.sqrt(-np.log(self.el))/self.cap]
             distances[k]=min(np.min(distances1),np.min(distances2))
             
         #define the shape parameter for the constrained point    
         cC=np.sqrt(-np.log(self.el))/(distances)         
         #Merging in one array
         self.c_P=np.hstack((self.c,cC))
         self.c_P[self.c_P>capp]=capp
         self.X_C_P=np.hstack((self.X_C,XBCC))
         self.Y_C_P=np.hstack((self.Y_C,YBCC))
         
        #If 3D
        if self.type=='3D':
            
         #Unpacking and eliminating double
         XBCC=np.hstack(self.XBC)
         YBCC=np.hstack(self.YBC)
         ZBCC=np.hstack(self.ZBC)
         BBB=np.unique(np.column_stack((roundDown(XBCC),roundDown(YBCC),roundDown(ZBCC))),axis=0)
         XBCC=BBB[:,0]
         YBCC=BBB[:,1]
         ZBCC=BBB[:,2]
         
         #Inizializing the distance of the constraint point
         distances=np.zeros(len(XBCC))
         
         #seeking for the nearest neighboor of the constrained point
         #in the constrained points and in the grid points         
         for k in np.arange(len(XBCC)):
             distances1=np.sqrt((self.X_C-XBCC[k])**2+(self.Y_C-YBCC[k])**2+(self.Z_C-ZBCC[k])**2)
             distances2=np.sqrt((np.delete(XBCC,k)-XBCC[k])**2+(np.delete(YBCC,k)-YBCC[k])**2+(np.delete(ZBCC,k)-ZBCC[k])**2)
             distances1=distances1[distances1>np.sqrt(-np.log(self.el))/self.cap]
             distances2=distances2[distances2>np.sqrt(-np.log(self.el))/self.cap]
             distances[k]=min(np.min(distances1),np.min(distances2))
             
         #define the shape parameter for the constrained point      
         cC=np.sqrt(-np.log(self.el))/distances
         
         #Merging in one array
         self.c_P=np.hstack((self.c,cC))
         self.c_P[self.c_P>capp]=capp
         self.X_C_P=np.hstack((self.X_C,XBCC))
         self.Y_C_P=np.hstack((self.Y_C,YBCC))
         self.Z_C_P=np.hstack((self.Z_C,ZBCC))
         return
     
    def pressure_computation(self,rcond=1e-13,method='fullcho',multilevel=0):
        
        """
          Compute the pressure
               
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param rcond: float (optional)
                      inverse of the maximum conditioning acceptable
          
          :param method: string (optional) 
                      default is 'fullcho' , other option are 'mixed' and 'fullpinv'. 'fullcho' uses only Cholensky
                      'fullpinv' uses only pseudo inverse and finally 'mixed' uses Cholensky for the variance matrix
                      and a pseudoinverse for the constraint part. Finally, 'solve' use np.linalg.solve on the complete matrix.             
          
          ----------------------------------------------------------------------------------------------------------------
        """
        
        #Saving the rcond
        self.rcond_P=rcond
        #If 2D
        if self.type=='2D':
            
         #Build the Boundary condition and matrix
         self.BC,self.MAT_CON=Boundary_Conditions(self.rho,self.mu,self.XBC,self.YBC,self.BCD,self.n,self.w_u,self.w_v,self.X_C_vel,self.Y_C_vel,self.cvel,self.X_C_P,self.Y_C_P,self.c_P,self.rcond_P)
         
         #Solve the Poisson equation
         self.w_P=Poisson_solver(self.rho,self.mu,self.X_C_P,self.Y_C_P,self.XG,self.YG,self.c_P,self.w_u,self.w_v,self.X_C_vel,self.Y_C_vel,self.cvel,self.MAT_CON,self.BC,self.rcond_P,method)
        
        if self.type=='3D':
         if self.model=='steady':
          #Build the Boundary condition and matrix
          self.BC,self.MAT_CON=Boundary_Conditions3D(self.rho,self.mu,self.XBC,self.YBC,self.ZBC,self.BCD,self.n,self.w_u,self.w_v,self.w_w,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.X_C_P,self.Y_C_P,self.Z_C_P,self.c_P,self.rcond_P)
          self.w_P=Poisson_solver3D(self.rho,self.mu,self.X_C_P,self.Y_C_P,self.Z_C_P,self.XG,self.YG,self.ZG,self.c_P,self.w_u,self.w_v,self.w_w,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.MAT_CON,self.BC,self.rcond_P,method)
         if self.model=='RANSI':  
           self.BC,self.MAT_CON=Boundary_Conditions3DRSI(self.rho,self.mu,self.XBC,self.YBC,self.ZBC,self.BCD,self.n,self.w_u,self.w_v,self.w_w,self.W_RSI,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.X_C_P,self.Y_C_P,self.Z_C_P,self.c_P,self.rcond_P)
           self.w_P=Poisson_solver3D(self.rho,self.mu,self.X_C_P,self.Y_C_P,self.Z_C_P,self.XG,self.YG,self.ZG,self.c_P,self.w_u,self.w_v,self.w_w,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.MAT_CON,self.BC,self.rcond_P,method)
         if self.model=='RANSA':  
           self.BC,self.MAT_CON=Boundary_Conditions3DRSA(self.rho,self.mu,self.XBC,self.YBC,self.ZBC,self.BCD,self.n,self.w_u,self.w_v,self.w_w,self.W_RSXX,self.W_RSXY,self.W_RSXZ,self.W_RSYY,self.W_RSYZ,self.W_RSZZ,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.X_C_P,self.Y_C_P,self.Z_C_P,self.c_P,self.rcond_P)
           self.w_P=Poisson_solver3DRSA(self.rho,self.mu,self.X_C_P,self.Y_C_P,self.Z_C_P,self.XG,self.YG,self.ZG,self.c_P,self.w_u,self.w_v,self.w_w,self.W_RSXX,self.W_RSXY,self.W_RSXZ,self.W_RSYY,self.W_RSYZ,self.W_RSZZ,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,self.cvel,self.MAT_CON,self.BC,self.rcond_P,method)
        del self.MAT_CON
        return
    
    def extrapolate_pressure(self,Fit_point):
        
        """
          Compute the pressure in the desired point              
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param Fit_point: list, 
                      list containing the point where to evaluate the pressure in such a form as Fit_point=[X,Y,Z] or [X,Y]. X can be also a matrix  (ex. from meshgrid).
          
          ----------------------------------------------------------------------------------------------------------------
          Returns
          -------

          :return P_fit: array, 
                    The pressure in the Fit_point . NB the shape of this array is the same of X
        """
        
        #If 2D
        if self.type=='2D':
         #Unpack the value
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         
         #Extrapolate pressure
         P_fit=Fit_RBF(self.w_P,self.X_C_P,self.Y_C_P,XFIT,YFIT,self.c_P,self.rcond_P)
         
        #If 3d
        if self.type=='3D':
         #Unpack the value
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         ZFIT=Fit_point[2]
         
         #Extrapolate pressure
         P_fit=Fit_RBF3D(self.w_P,self.X_C_P,self.Y_C_P,self.Z_C_P,XFIT,YFIT,ZFIT,self.c_P,self.rcond_P)
         
        return P_fit
    
    def extrapolate_RS(self,Fit_point):
        
        """
          Compute the Reynolds stresses in the desired point               
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param Fit_point: list, 
                      list containing the point where to evaluate the pressure in such a form as Fit_point=[X,Y,Z] or [X,Y]. X can be also a matrix  (ex. from meshgrid).
          
          ----------------------------------------------------------------------------------------------------------------
          Returns
          -------

          :return:list or array, 
                    The Reynolds in the Fit_point . NB the shape of this array is the same of X
        """
        
        if self.type=='3D':
         if self.model=='RANSI':
         #Unpack the value
          XFIT=Fit_point[0]
          YFIT=Fit_point[1]
          ZFIT=Fit_point[2]
         
          SHAPE=np.shape(XFIT)
         #Extrapolate pressure
         
          RSI=PHI3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),ZFIT.reshape(-1),self.cvel,self.rcond_vel)@self.W_RSI
          return RSI
         if self.model=='RANSA':
         #Unpack the value
          XFIT=Fit_point[0]
          YFIT=Fit_point[1]
          ZFIT=Fit_point[2]
         
          SHAPE=np.shape(XFIT)
         #Extrapolate pressure
          phi=PHI3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),ZFIT.reshape(-1),self.cvel,self.rcond_vel)
          RSXX=phi@self.W_RSXX
          RSXX=RSXX.reshape(SHAPE)
          RSYY=phi@self.W_RSYY
          RSYY=RSYY.reshape(SHAPE)
          RSZZ=phi@self.W_RSZZ
          RSZZ=RSZZ.reshape(SHAPE)
          RSXY=phi@self.W_RSXY
          RSXY=RSXY.reshape(SHAPE)
          RSXZ=phi@self.W_RSXZ
          RSXZ=RSXZ.reshape(SHAPE)
          RSYZ=phi@self.W_RSYZ
          RSYZ=RSYZ.reshape(SHAPE)
          return [RSXX,RSYY,RSZZ,RSXY,RSXZ,RSYZ]
    
    def extrapolate_divergence(self,Fit_point):
        """
          Compute the pressure in the desired point              
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param Fit_point: list, 
                      list containing the point where to evaluate the pressure in such a form as Fit_point=[X,Y,Z] or [X,Y]. X can be also a matrix  (ex. from meshgrid).
          
          ----------------------------------------------------------------------------------------------------------------
          Returns
          -------

          :return U: array, 
                    The divergence in the Fit_point . NB the shape of this array is the same of X
        """
        
        #If 2D
        if self.type=='2D':
         
         #Unpack the value
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         
         #Estrapolating
         SHAPE=np.shape(XFIT)
         DX=Der_RBF_X(self.X_C_vel,self.Y_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),self.cvel,self.rcond_vel)
         DY=Der_RBF_Y(self.X_C_vel,self.Y_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),self.cvel,self.rcond_vel)
         U1=DX.dot(self.w_u)+DY.dot(self.w_v)
         U=U1.reshape(SHAPE)
         
        #If 3D
        if self.type=='3D':
            
         #Unpack the value
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         ZFIT=Fit_point[2]
         
         #Estrapolating
         SHAPE=np.shape(XFIT)
         DX=Der_RBF_X3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),ZFIT.reshape(-1),self.cvel,self.rcond_vel)
         DY=Der_RBF_Y3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),ZFIT.reshape(-1),self.cvel,self.rcond_vel)
         DZ=Der_RBF_Z3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT.reshape(-1),YFIT.reshape(-1),ZFIT.reshape(-1),self.cvel,self.rcond_vel)
         U1=DX.dot(self.w_u)+DY.dot(self.w_v)+DZ.dot(self.w_w)
         U=U1.reshape(SHAPE)    
         
        return U
    
    def extrapolate_velocities(self,Fit_point):
        """
          Compute the velocity field in the desired point
               
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param Fit_point: list, 
                      list containing the point where to evaluate the pressure in such a form as Fit_point=[X,Y,Z] or [X,Y]. X can be also a matrix  (ex. from meshgrid).
          
          ----------------------------------------------------------------------------------------------------------------
          Returns
          -------

          :return fit: array, 
                    The velocities in the Fit_point .Given as [u,v,w]. NB the shape of u (for example) is the same of X
        """
        
        #If 2D
        if self.type=='2D':
            
         #Unpack the values
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         
         #Estrapolating
         u_fit,v_fit=Fit_vel2D(self.w_u,self.w_v,self.X_C_vel,self.Y_C_vel,XFIT,YFIT,self.cvel,self.rcond_vel)
         fit=[u_fit,v_fit]
         
        #If 3D 
        if self.type=='3D':
            
         #Unpack the values   
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         ZFIT=Fit_point[2]
         
         #Estrapolating
         u_fit,v_fit,w_fit=Fit_vel3D(self.w_u,self.w_v,self.w_w,self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT,YFIT,ZFIT,self.cvel,self.rcond_vel)
         fit=[u_fit,v_fit,w_fit]
         
         
        return fit
    
    
    def extrapolate_vorticity(self,Fit_point):
        """
          Compute the vorticity in the desired point
               
          
          ----------------------------------------------------------------------------------------------------------------
          Parameters
          ----------
          :param Fit_point: list, 
                      list containing the point where to evaluate the pressure in such a form as Fit_point=[X,Y,Z] or [X,Y]. X can be also a matrix  (ex. from meshgrid).
          
          ----------------------------------------------------------------------------------------------------------------
          Returns
          -------

          :return fit: array, 
                    if 3D the three component are stacked otherwise give an array
        """
        
        #If 2D
        if self.type=='2D':
            
         #Unpack the values
         XFIT=Fit_point[0]
         YFIT=Fit_point[1]
         
         #Estrapolating
         DER_X=Der_RBF_X(self.X_C_vel,self.Y_C_vel,XFIT,YFIT,self.cvel,self.rcond_vel)
         VX=DER_X.dot(self.w_v)
         del DER_X
    
         #the derivatives of velocity are evaluated
         DER_Y=Der_RBF_Y(self.X_C_vel,self.Y_C_vel,XFIT,YFIT,self.cvel,self.rcond_vel)
         UY=DER_Y.dot(self.w_u)
         del DER_Y
         return VX-UY
         
        #If 3D 
        if self.type=='3D':
            
         #Unpack the values   
            XFIT=Fit_point[0]
            YFIT=Fit_point[1]
            ZFIT=Fit_point[2]
             
            DER_X=Der_RBF_X3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT,YFIT,ZFIT,self.cvel,self.rcond_vel)#Differentation matrix
            VX=DER_X.dot(self.w_v)
            WX=DER_X.dot(self.w_w)
            del DER_X
            
            #the derivatives of velocity are evaluated
            DER_Y=Der_RBF_Y3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT,YFIT,ZFIT,self.cvel,self.rcond_vel)#Differentation matrix
            UY=DER_Y.dot(self.w_u)
            WY=DER_Y.dot(self.w_w)
            del DER_Y
            
            #the derivatives of velocity are evaluated
            DER_Z=Der_RBF_Z3D(self.X_C_vel,self.Y_C_vel,self.Z_C_vel,XFIT,YFIT,ZFIT,self.cvel,self.rcond_vel)#Differentation matrix
            UZ=DER_Z.dot(self.w_u)
            VZ=DER_Z.dot(self.w_v)
            del DER_Z
            
         
        return np.vstack((WY-VZ,UZ-WX,VX-UY)).T    

