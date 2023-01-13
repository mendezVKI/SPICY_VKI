# -*- coding: utf-8 -*-
"""
Latest update on Thu Jan 12 17:56:06 2023

@author: mendez, ratz, sperotto
"""


import numpy as np # used in all computations

# these functions are used for the clutering and collocation
from sklearn.neighbors import NearestNeighbors

# Note: there is a warning from kmeans when running on windows.
# This should fix it
import warnings
warnings.filterwarnings('ignore')

from sklearn.cluster import MiniBatchKMeans






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
    def __init__(self,data,grid_point,model,basis='exp',ST=None):
        """
        Initialization of an instance of the spicy class.
             
        # The input parameters are 
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param model: string
                   This defines the model. Currently, SPICY supports 4 models:
                       1. 'scalar', to regress a scalar quantity.
                           This is implemented both in 2D and 3D.
                       2. 'laminar', to regress the velocity field without turbulence modeling.
                            This is implemented both in 2D and 3D.
                       3.  'RANSI', to regress a velocity field with a RANS model assuming isotropic Reynolds stresses (hence mean(u'**2)) is the only extra expected quantity.
                            This must be provided as the fourth entry of 'data', which becomes [u,v,w,uu].
                            NOTE: Currently, RANSI is only implemented in 3D.
                       4.  'RANSA', to regress a velocity field with a RANS model without assuming isotropic Reynolds stresses.
                            this becomes [uu, vv, uv] in 2D and [uu, vv, ww, uv, uw, vw] in 3D.
                            NOTE: Currently, RANSA is only implemented in 3D.                            
        :param data: list
                    Is a list of arrays containing [u] if the model is scalar,
                    [u,v] for a 2D vector field and [u,v,w] for a 3D field.
                    
        :param grid_point: list
                    Is a list of arrays containing the grid point [XG,YG,ZG] in 3D and [XG,YG] in 2D.   
        :param basis: string
                    This defines the basis. Currently, the two options are:
                     1. 'gauss', i.e. Gaussian RBFs exp(-c_r**2*d(x))
                     2. 'c4', i.e. C4 RBfs (1+d(x+)/c_r)**5(1-d(x+)/c_r)**5
        
        :param ST: list
                    Is a list of arrays collecting Reynolds stresses. This is empty if the model is 'scalar' or 'laminar'.
                    If the model is RANSI, it contains [uu']. 
                    If the model is RANSA, it contains [uu, vv, uv] in 2D and [uu, vv, ww, uv, uw, vw] in 3D.                                                   
                            
        ----------------------------------------------------------------------------------------------------------------
        Attributes
        ----------
        
        XG, YG, ZG: coordinates of the point in which the data is available
        u : function to learn or u component in case of velocity field
        v: v component in case of velocity field (absent for scalar)
        w: w component in case of velocity field (absent for scalar)
        
        RSI: Reynolds stress in case of isotropic flow (active for RANSI)
        
        [...] TODO Manuel please finalize this documentation (also including the methods)
        
        """
        
        # assign inputs
        self.model=model 
        self.basis=basis
        
        # Initialize attributes that will be needed later:
        self.X_C=[]
        self.Y_C=[]
        self.c_k=[]
        self.d_k=[]   
        
        
                
        if len(grid_point)==2:
            self.type='2D'
            self.XG=grid_point[0]
            self.YG=grid_point[1]
            
        if len(grid_point)==3:
            self.type='3D'
            self.XG=grid_point[0]
            self.YG=grid_point[1]
            self.ZG=grid_point[2] 
            
        
        if model=='scalar':
            self.u=data[0]     
            
            
        if model=='laminar' and len(grid_point)==2:
            self.u=data[0]
            self.v=data[1]
            
        if model=='laminar' and len(grid_point)==3:
            self.u=data[0]
            self.v=data[1]
            self.w=data[2]
        
        if model=='RANSI' and len(grid_point)==2:
            raise ValueError('RANS models are currently only implemented in 3D')
        
            
        if model=='RANSA' and len(grid_point)==2:
            raise ValueError('RANS models are currently only implemented in 3D')
            
        if model=='RANSI' and len(grid_point)==3:
            self.RSI=data[3]
            self.Constr_RS=np.array([])
            self.XCONRS=np.array([])
            self.YCONRS=np.array([])
            self.ZCONRS=np.array([])
              
        if model=='RANSA' and len(grid_point)==3:            
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
    




# #%% 1. Constraints.

# We have two sorts of constraints: scalar and vector.
# scalar apply to model=scalar and to the poisson solvers.
# vector apply to all the other models.

# the scalar ones include: Dirichlet and Neuman.
# the vector one include: Dirichlet, Neuman and Div free.


    def scalar_constraints(self,DIR=[],NEU=[]):
        """         
        # The input parameters are 
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
        :param DIR: list
                   This contains the info for the Dirichlet conditions.
                   if the model is 2D, then this has [X_D,Y_D,c_D].
                   if the model is 3D, then this has [X_D,Y_D,Z_D,c_D].
              
        Here X_D, Y_D , Z_D are the coordinates of the poins where the value c_D
        is set.
        
        :param NEU: list
                   This contains the info for the Neuman conditions.
                   if the model is 2D, then this has [X_N,Y_N,n_x,n_y,c_N].
                   if the model is 3D, then this has [X_N,Y_N,Z_n,n_x,n_y,n_z,c_N].
                   
        Here X_N, Y_N , Z_N are the coordinates of the poins where the value c_N
               is set for the directional derivative along the normal direction n_x,n_y,n_z
                 
               
        [...] TODO Manuel please finalize this documentation (including the other attributes)
        It could also be worth raising error.
        for example, if len(DIR) is different than 3 when type is 2D we have a problem.
        
        NOTE: there is no check on the possible (dangerous overlapping of conditions).
        Therefore, at the moment, one might put both Neuman and Dirichlet conditions
        at the same points. This is of course a terrible idea.
        TODO in future release: if both D and N conditions are given in at the same points
        ( or to close?) then only one of them (e.g. D) is considered
        
        
        """
    
        # Set the Dirichlet conditions
        if not DIR:
          print('No D conditions')   
        else:
         #Check if we have 2D or a 3D problem.
         if len(DIR)==3: # This means we have a 2D problem
           self.n_D=len(DIR[0])
           self.X_D=DIR[0]
           self.Y_D=DIR[1]
           self.c_D=DIR[2]
         else:
           self.n_D=len(DIR[0])
           self.X_D=DIR[0]
           self.Y_D=DIR[1]
           self.Z_D=DIR[2]
           self.c_D=DIR[3]
         print(str(self.n_D)+' D conditions assigned') 
  
        # Set the Neuman conditions
       
        if not NEU:
          print('No N conditions')            
        else: 
          #Check if we have 2D or a 3D problem.
          if len(NEU)==3: # This means we have a 2D problem
            self.n_N=len(NEU[0])
            self.X_N=NEU[0]
            self.Y_N=NEU[1]
            self.n_x=NEU[2]
            self.n_y=NEU[3]
            self.c_N=NEU[4]
          else:
            self.n_N=len(NEU[0])
            self.X_N=NEU[0]
            self.Y_N=NEU[1]
            self.Z_N=NEU[1]            
            self.n_x=NEU[2]
            self.n_y=NEU[3]
            self.n_z=NEU[3]            
            self.c_N=NEU[4]           
          print(str(self.n_N)+' N conditions assigned')
    
        return

# #%% 2. Clustering (this does not depend on the model, but only on the dimension).
    def clustering(self,n_K,r_mM=[0.01,0.3],eps_l=0.7):
      """
        This function defines the collocation of a set of RBFs using the multi-level clustering
        described in the article
         
       #The input parameters are 
        ----------------------------------------------------------------------------------------------------------------
        Parameters
        ----------
      :param n_K: list
              This contains the n_k vector in eq 33. if n_K=[4,10], it means that the clustering
              will try to have a first level with RBFs whose size seeks to embrace 4 points, while the 
              second level seeks to embrace 10 points, etc.
              The length of this vector automatically defines the number of levels.
      
      :param d_mM: list
              This contains the minimum and the maximum RBF's radius. This is defined as the distance from the
              collocation point at which the RBF value is 0.5.
              
      :param eps_l: float
              This is the value that a RBF will have at its closest neighbour. It is used to define the shape
              factor from the clustering results.
                 
      """
      # Check if we are dealing with a 2D or a 3D case. For the moment, I implement only 2D.
      if self.type=='2D':
        # reassign the key variable (to avoid using 'self' below)
        XG=self.XG; YG=self.YG ; n_p=len(XG)
        # Stack the coordinates in a matrix:
        D=np.column_stack((XG,YG))
        n_l=len(n_K)  # Number of levels
        
        for l in range(n_l):
         Clust=int(np.ceil(n_p/n_K[l])) # define number of clusters
         #initialize the cluster function
         model=MiniBatchKMeans(n_clusters=Clust, random_state=0)    
         # Run the clustering and return the indices:
         y_P=model.fit_predict(D)
         #obtaining the centers of the points
         Centers=model.cluster_centers_
         
         # Get the nearest neighbour of each center:
         nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Centers)
         distances, indices = nbrs.kneighbors(Centers)
         sigma1=distances[:,1]
        
         # count how manye points are in each cluster:
         #count=np.bincount(y_P,minlength=Clust)
         
         # Fix the minimum distance if a cluster has less than min(n_K) or if
         # the distance 
        # sigma1[sigma1==0]=np.mean(sigma1[sigma1!=0])   
        # sigma1[count<np.min(n_K)]=np.mean(sigma1[sigma1!=0])  
         # Pre-assign the collocation points
         X_C1=Centers[:,0]
         Y_C1=Centers[:,1]
         
         # Assign the results to a vector of collocation points
         if l==0: # If this is the first layer, just assign:
            X_C=X_C1 
            Y_C=Y_C1 
            sigma=sigma1 
         else:
            X_C=np.hstack((X_C,X_C1))
            Y_C=np.hstack((Y_C,Y_C1))
            sigma=np.hstack((sigma,sigma1))
         print('Clustering level '+str(l)+' completed')
         
      # We conclude with the computation of the shape factors.
      # These depends on the type of RBF
         if self.basis =='gauss':
           # Set the max and min values of c:  
           c_min=1/(2*r_mM[1])*np.sqrt(np.log(2))
           c_max=1/(2*r_mM[0])*np.sqrt(np.log(2))
           # compute the c 
           c_k=np.sqrt(-np.log(eps_l))/sigma
           c_k[c_k<c_min]=c_min; c_k[c_k>c_max]=c_max
           # for plotting purposes, we store also the diameters
           d_k=1/c_k*np.sqrt(np.log(2))
           
         # Assign the output
         self.X_C=X_C
         self.Y_C=Y_C
         self.c_k=c_k
         self.d_k=d_k
           
      return


# def plot_RBFs:
    
    
#     return


# #%% 3. Assembly A, B, C(this depends on everything)

# def Assembly:
    
#     return








    
    

