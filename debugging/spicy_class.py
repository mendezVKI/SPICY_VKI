# -*- coding: utf-8 -*-
"""
Latest update on Thu Jan 12 17:56:06 2023

@author: mendez, ratz, sperotto
"""


import numpy as np # used in all computations

# these functions are used for the clutering and collocation
from sklearn.neighbors import NearestNeighbors
# Function for the k means clusering
from sklearn.cluster import MiniBatchKMeans

# Note: there is a warning from kmeans when running on windows.
# This should fix it
import warnings
warnings.filterwarnings('ignore')


# Matplotlib for the plotting functions:
import matplotlib.pyplot as plt 


# function useful for computing smallsest and largest eig:
from scipy.sparse.linalg import eigsh
# we use scipy linalg for cholesky decomposition, solving linear systems etc
from scipy import linalg




#%%%%% SPICY CLASS BODY: $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    
class spicy:
    def __init__(self,data,grid_point,model,basis='exp',ST=None,Mem_Sav='y'):
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
                            
        :param Mem_Sav: string
                    If 'y', a 'memory saving approach' is followed through all functions.
                    This means no matrix is stored in the RAM unless it is being used by a specific function.
                    This significantly reduces the memory usage but slows down the solver, since it must often save and load things
                    from the local driver. Moreover, temporary data will be produced on your local path.
                    All the relevant codes will be ported into two
                                                  
                    
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
    

#%% 1. Constraints.

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
        
        NOTE: there is no check on the possible (dangerous) overlapping of conditions.
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
      
      :param r_mM: list
              This contains the minimum and the maximum RBF's radiuses. This is defined as the distance from the
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
        
         #count how manye points are in each cluster:
         #count=np.bincount(y_P,minlength=Clust)
         
         # Fix the minimum distance if a cluster has less than min(n_K) or if
         # the distance 
         #sigma1[sigma1==0]=np.mean(sigma1[sigma1!=0])   
         #sigma1[count<np.min(n_K)]=np.mean(sigma1[sigma1!=0])  
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


    def plot_RBFs_2D(self):
     """
     This function provides a quick plot to let the user analyze 
     how the RBFs have been placed. 
     Note inputs are needed and no variable is returned
    
     """
   
     if self.model=='scalar':
       try:  
        fig, axs = plt.subplots(1, 2)
        # First plot is the RBF distribution
        axs[0].set_title("RBF Collocation")
        for i in range(0,len(self.X_C),1):
           circle1 = plt.Circle((self.X_C[i], self.Y_C[i]), self.d_k[i]/2, 
                                 fill=True,color='g',edgecolor='k',alpha=0.2)
           axs[0].add_artist(circle1)  
        axs[0].scatter(self.XG,self.YG,c=self.u)

        if hasattr(self,'X_D') and hasattr(self,'X_N'):
         axs[0].plot(self.X_D, self.Y_D,'ro')
         axs[0].plot(self.X_N, self.Y_N,'bs')
        elif hasattr(self,'X_D'):              
         axs[0].plot(self.X_D, self.Y_D,'ro')
        elif hasattr(self,'X_N'):              
         axs[0].plot(self.X_N, self.Y_N,'bs')   
        else:
         print('No constraints set.')    
        
        # second plot is the distribution of diameters:
        axs[1].stem(self.d_k/2)
        axs[1].set_xlabel('Basis index')
        axs[1].set_ylabel('Diameter') 

        axs[1].set_title("Distribution of diameters")
        fig.tight_layout()
       
       except:
        raise ValueError('Problems in plotting. Set constraints and cluster!')   
    
     else:
       print('plotting available only for scalar model. Work in progress')  
       return


# #%% 3. Assembly A, B, b_1, b_2  (this depends on everything)

    def Assembly_Poisson_2D(self,source_terms,n_hb=5,extra_RBF='y',eps_lb=0.2):
       """
       This function assembly the matrices A, B, C, D from the paper.
       TODO. Currently implemented only for model='scalar'
   
       #The input parameters are 
       ----------------------------------------------------------------------------------------------------------------
       Parameters
       ----------
       :param source_terms: array
          This is relevant only in the 'scalar' model. 
          This vector contains the values for the source term on all the given points (term s in eq 27).
          To solve the Laplace equation, it should be a vector of zeros.
          To solve the Poisson equation for the pressure, this is the RHS of eq.21.
          In any case, the specification of the RHS is done outside the assmebly function.
            
       :param n_hb: string (currently not active)
          When solving the Poisson equation, global basis elements such as polynomials or series
          expansions are of great help. This is evident if one note that the eigenfunctions of 
          the Laplace operator are harmonics. 
          In a non-homogeneous problem, once could homogenize the basis. This will be proposed for the next relase
          (which will align with the paper of Manuel). The idea is the following: if the homogeneization is well done and
          the basis is well chosen, then we will not need constraints for these extra terms of the basis.
                  
          For the moment, we let the user introduce the number of extra_basis. 
          These will be sine and cosine bases, which are orthogonal in [-1,1].
          In 1D, they are defined as : sines_n=np.sin(2*np.pi*(n)*x); cos_n=np.cos(np.pi/2*(2*n+1)*x)
          Given n_hb, we will have that the first n_hb are sines the last n_hb will be cosines.
          This defines the basis phi_h_n, with n an index from 0 to n_hb**4 in 2D.
                    
          In 2D, assuming separation of variables, we will take phi_h_nm=phi_n(x)*phi_m(y).
          Similarly, in 3D will be phi_nmk=phi_n(x)*phi_m(y)*phi_k(z).
          For stability purposes, the largest tolerated value at the moment is 10!.
          
          For an homogeneous problem, the chosen basis needs no constraints.          
   
       :param extra_RBF: string (currently not active) 
          In presence of non-homogeneous conditions, it might help adding
          additional rbfs on the locations where the boundaries are to be enforced.
          If this parameter is 'y', SPICY will do that. Otherwise, it will not add extra bases.
    
       :param eps_lb: float. 
       
   
        """   
       self.n_hb=n_hb
       
       if self.model=='scalar':
        # To facilitate debugging, I reassign the variables here:
        XG=self.XG; YG=self.YG; X_C=self.X_C; Y_C=self.Y_C; c_k=self.c_k  
               
        # We start with a check on the BCs. In all points in which we do not have
        # homogeneous conditions, we should consider boundary points as collocations points.
        # if extra_RBF='y' we check wheter we have non-homogeneous conditions. Then, in these 
        # points, we put a RBF with shape factor computed in such a way that
        # its value on the nearest boundary point is eps_l_b.

    
        # 1. Check if we have constraints Dirichlet/Neuman conditions
        # Check c_D
        if hasattr(self,'c_D'): 
         n_D=len(self.c_D)
         X_D=self.X_D; Y_D=self.Y_D
        else:
         n_D=0
        # Check c_N
        if hasattr(self,'c_N'):
         n_N=len(self.c_N)
        else:
         n_N=0
         
        # Approach 1: we build A, B, b1, b2 as in the article from Sperotto
        L_RBF=Laplacian_2D_RBF(XG,YG,X_C,Y_C,c_k)
        L_H=Laplacian_2D_H(XG,YG,n_hb)
        L=np.hstack((L_H,L_RBF)) 
        # Then A and b1 are :
        self.A=2*L.T@L; self.b_1=2*L.T.dot(source_terms)  
        
        # The constraint matrix depends on n_D an n_N
        if n_D !=0 and n_N==0: # you have Dirichlet and no Neuman
         self.B=np.hstack((Phi_2D_harm(X_D, Y_D, n_hb),Phi_2D_RBF(X_D, Y_D, X_C, Y_C, c_k))).T  
         self.b_2=self.c_D
        elif n_D ==0 and n_N !=0: # you have Neuman and no Dirichlet
         print('you have Dirichlet and no Neuman')          
        elif n_D !=0 and n_N !=0: # You have both N and D
         print('you have Dirichlet and no Neuman')
        else: # You have no constraints
         print('you have Dirichlet and no Neuman')
        

    
        # Get the Laplacian at the points (XG,YG) from the collocation points
        # at (X_C,Y_C) and considering additional n_hb homogeneous spectral basis element.       

        
        # Get the constraint matrices:
                  
       else:
        print('Assembly only build for the scalar function') 
    
       return

# Solver using the Shur complement


    def Solve_2D(self,K_cond=1e12):
     """
     This function solves the constrained quadratic problem A, B, b_1, b_2.
     TODO. Currently implemented only for model='scalar'

     #The input parameters are 
     ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param K_cond: float
       This is the regularization parameter. It is fixing the condition number
       The estimation is based such that the regularize matrix has the condition
        number k_cond. For this, we compute the max and the min eigenvalue.
     """   
 
     # Assign variables for debugging purposes
     A=self.A; B=self.B ; b_1=self.b_1; b_2=self.b_2
     
     # Step 1: Regularize the matrix A
     try:
      lambda_m = eigsh (A , 1 , sigma =0.0,return_eigenvectors=False) # smallest eigenvalue
      lambda_M=eigsh(A,1,return_eigenvectors=False) # Largest eigenvalue
      alpha=(lambda_M-K_cond*lambda_m)/(K_cond-1)
     except:
      lambda_M=eigsh(A,1,return_eigenvectors=False) # Largest eigenvalue
      alpha=(lambda_M)/(K_cond-1)
      print('Warning, lambda_m could not be computed in A')   
     A= A+alpha*np.eye(np.shape(A)[0])
     print('Matrix A regularized')
     
     # Step 2: Cholesky Decomposition of A    
     L_A,low=linalg.cho_factor(A,overwrite_a=True,check_finite=False,lower=True)
     # Step 3: Solve for N
     N=linalg.cho_solve((L_A,low),B,check_finite=False)
        
     
     # Step 4: prepare M 
     M=N.T@B
     # Step 5: Regularize M
     try:
      lambda_m = eigsh (M , 1 , sigma =0.0,return_eigenvectors=False) # smallest eigenvalue
      lambda_M=eigsh(M,1,return_eigenvectors=False) # Largest eigenvalue
      alpha=(lambda_M-K_cond*lambda_m)/(K_cond-1)
     except:
      lambda_M=eigsh(M,1,return_eigenvectors=False) # Largest eigenvalue
      alpha=(lambda_M)/(K_cond-1)
      print('Warning, lambda_m could not be computed in M')
     
     M= M+alpha*np.eye(np.shape(M)[0])
     print('Matrix M computed and regularized')
     
     # Step 6: get the chol factor of M    
     L_M,low=linalg.cho_factor(M,overwrite_a=True,check_finite=False,lower=True)

     # Step 7: Solve the system for lambda    
     b2star=N.T.dot(b_1)-b_2
     self.lam=linalg.cho_solve((L_M,low),b2star,check_finite=False)
     print('Lambdas computed')

     # Step 8: Solve for w.
     b1_star=b_1-B.dot(self.lam)
     self.w=linalg.cho_solve((L_A,low),b1_star,check_finite=False)
     print('w computed')

     # You could estimate the error in the solutions:
     # err_w=np.linalg.norm(A.dot(self.w)+B.dot(self.lam)-b_1)    
     # err_lam=np.linalg.norm(B.T.dot(self.w)-b_2)    

     return 


# Here is a function to compute the solution on an arbitrary set of points
# XG, YG. We take w, lam from the solution, X_C, Y_C, c_k from the clustering.

    def Get_Sol_2D(self,XP,YP):
     """
     This function plots the solution (w,lam) on an arbitrary set of points (XG,YG),
         Get the basis matrix at the points (XP,YP) from RBFs at the collocation points
          at (X_C,Y_C), having shape factors c_k.       
         The output is a matrix of side (n_p) x (n_c)
        
     TODO. Write a better help here ( with parameters etc)

     #The input parameters are 
     ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------
     :param XP,YP: 
         
     """   
 
     # Assign variables for debugging purposes
     X_C=self.X_C; Y_C=self.Y_C ; c_k=self.c_k; w=self.w; lam=self.lam
     n_hb=self.n_hb;
     
     # Form the matrix
     Phi=np.hstack((Phi_2D_harm(XP, YP, n_hb),Phi_2D_RBF(XP, YP, X_C, Y_C, c_k)))  
     U_P=Phi.dot(w)

     return U_P





#%% Utilities function %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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


def Laplacian_2D_RBF(XG,YG,X_C,Y_C,c_k):
    # Get the Laplacian at the points (XG,YG) from RBFs at the collocation points
    # at (X_C,Y_C), having shape factors c_k, 
    
    # This is the contribution of the RBF part
    # The number of RBF bases (n_b) and the number of points (n_p) are:
    n_b=len(X_C); n_p=len(XG)    
    Lap_RBF=np.zeros((n_p,n_b)); # RBF portions
    
    # What comes next depends on the type of chosen RBF
    for r in range(n_b):
     gaussian=np.exp(-c_k[r]**2*((X_C[r]-XG)**2+(Y_C[r]-YG)**2))
     Partial_xx=4*c_k[r]**4*(X_C[r]-XG)**2*gaussian-2*c_k[r]**2*gaussian
     Partial_yy=4*c_k[r]**4*(Y_C[r]-YG)**2*gaussian-2*c_k[r]**2*gaussian
     Lap_RBF[:,r]=Partial_xx+Partial_yy
     # debugging: you might want to have a look:
     # plt.scatter(XG,YG,c=Partial_yy)    
    
    return Lap_RBF
    

def Laplacian_2D_H(XG,YG,n_hb):
    # Get the Laplacian at the points (XG,YG) from n_hb homogeneous spectral
    # basis element. The output is a matrix of side (n_p) x (n_c+n_hb**4)

    # number of points
    n_p=len(XG)    
       
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Lap_H=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y
    
    for i in range(n_hb):
     for j in range(n_hb):
       for m in range(n_hb):
          for q in range(n_hb):            
            k_x_i=2*np.pi*(i+1) # This goes with sines
            k_x_j=np.pi/2*(2*j+1) # This goes with cosines
            k_y_m=2*np.pi*(m+1) # This goes with sines
            k_y_q=np.pi/2*(2*q+1) # This goes with cosines
            # To take the differentiation, we use automatic diff style:
            sin_k_i_x=np.sin(k_x_i*XG); cos_k_i_x=np.sin(k_x_i*XG)
            cos_k_j_x=np.cos(k_x_j*XG); sin_k_j_x=np.sin(k_x_j*XG)
            sin_k_m_y=np.sin(k_y_m*YG); cos_k_m_y=np.cos(k_y_m*YG)
            cos_k_q_y=np.cos(k_y_q*YG); sin_k_q_y=np.sin(k_y_q*YG)
                        
            # Compute the derivatives of the harmonic basis sin_k_i_x
            phi_ijmq_xx=-sin_k_m_y*cos_k_q_y*(2*k_x_i*k_x_j*cos_k_i_x*sin_k_j_x+
                                             (k_x_j**2+k_x_i**2)*sin_k_i_x*cos_k_j_x)
            
            phi_ijmq_yy= -sin_k_i_x*cos_k_j_x*(2*k_y_m*k_y_q*cos_k_m_y*sin_k_q_y+
                                             (k_y_q**2+k_y_m**2)*sin_k_m_y*cos_k_q_y)
            # Assign the column of the Laplacian
            Lap_H[:,count]=phi_ijmq_xx+phi_ijmq_yy
            count+=1  
            
    # # # Here's how to see these        
    # plt.scatter(XG,YG,c=Lap_H[:,1])   
     
    # L=np.hstack((Lap_H,Lap_RBF))

    
    return Lap_H



def Phi_2D_RBF(XG,YG,X_C,Y_C,c_k):
    """
    Get the basis matrix at the points (XG,YG) from RBFs at the collocation points
     at (X_C,Y_C), having shape factors c_k.       
    The output is a matrix of side (n_p) x (n_c)
    """
    # This is the contribution of the RBF part
    n_b=len(X_C); n_p=len(XG)
    Phi_RBF=np.zeros((n_p,n_b))
    
    for r in range(n_b):
       gaussian=np.exp(-c_k[r]**2*((X_C[r]-XG)**2+(Y_C[r]-YG)**2))
       Phi_RBF[:,r]=gaussian
    # debugging: you might want to have a look:
    # plt.scatter(XG,YG,c=Phi_RBF[:,4]) 
    
       # # # Here's how to see these        
    # plt.scatter(XG,YG,c=Phi_H[:,1])   
     
    
    return Phi_RBF


def Phi_2D_harm(XG,YG,n_hb):
    # Get the basis matrix at the points (XG,YG) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4+1)
    
    # Get the number of points
    n_p=len(XG)
    
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Phi_H=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y

    for i in range(n_hb):
     for j in range(n_hb):
       for m in range(n_hb):
          for q in range(n_hb):            
            k_x_i=2*np.pi*(i+1) # This goes with sines
            k_x_j=np.pi/2*(2*j+1) # This goes with cosines
            k_y_m=2*np.pi*(m+1) # This goes with sines
            k_y_q=np.pi/2*(2*q+1) # This goes with cosines
            # To take the differentiation, we use automatic diff style:
            sin_k_i_x=np.sin(k_x_i*XG); 
            cos_k_j_x=np.cos(k_x_j*XG); 
            sin_k_m_y=np.sin(k_y_m*YG); 
            cos_k_q_y=np.cos(k_y_q*YG); 
                                    
            # Assign the column of Phi_H
            Phi_H[:,count]=sin_k_i_x*cos_k_j_x*sin_k_m_y*cos_k_q_y
            count+=1  
    
    # # # Here's how to see these        
    # plt.scatter(XG,YG,c=Phi_H[:,1])   
     
      
    return Phi_H

##############  Derivative operators in 2D for RBF ###########################


def Phi_RBF_2D_x(XG,YG,X_C,Y_C,c_k):
    """
    Create the derivatives along x, Phi_x, for the RBF bases with collocation points (X_C,Y_C) and 
    shape factors c_k, computed on the points (XG,YG)
    """
    # number of bases (n_b) and points (n_p)
    n_b=len(X_C); n_p=len(XG)
    # Initialize the matrix
    Phi_RBF_x=np.zeros((n_p,n_b))
    
    for r in range(n_b):
      gaussian=np.exp(-c_k[r]**2*((X_C[r]-XG)**2+(Y_C[r]-YG)**2))
      Phi_RBF_x[:,r]=-2*c_k[r]**2*(X_C[r]-XG)*gaussian
    
    return Phi_RBF_x


def Phi_RBF_2D_y(XG,YG,X_C,Y_C,c_k):
    """
    Create the derivatives along y, Phi_y, for the RBF bases with collocation points (X_C,Y_C) and 
    shape factors c_k, computed on the points (XG,YG)
    """
    # number of bases (n_b) and points (n_p)
    n_b=len(X_C); n_p=len(XG)
    # Initialize the matrix
    Phi_RBF_y=np.zeros((n_p,n_b))
  
    for r in range(n_b):
      gaussian=np.exp(-c_k[r]**2*((X_C[r]-XG)**2+(Y_C[r]-YG)**2))
      Phi_RBF_y[:,r]=-2*c_k[r]**2*(Y_C[r]-YG)*gaussian
  
    return Phi_RBF_y


def Phi_2_D_N_RBF(X_N,Y_N,X_C,Y_C,c_K,n_x,n_y):
    """
    Create the Phi_n operator for the RBF bases with collocation points (X_C,Y_C) and 
    shape factors c_k, computed on the points (X_N,Y_N) and 
    given the normal components (n_x,n_y). This is useful for defining Neuman conditions
    """
    Phi_N=Phi_RBF_2D_x(X_N,Y_N,X_C,Y_C,c_K)*n_x+Phi_RBF_2D_y(X_N,Y_N,X_C,Y_C,c_K)*n_y  
    
    return Phi_N

##############  Derivative operators in 2D for harmonics ###########################


def Phi_H_2D_x(XG,YG,n_hb):
    """
    Create the derivatives along x, Phi_x, for the n_hb harmonic bases, 
    computed on the points (XG,YG)
    """
    # Get the number of points
    n_p=len(XG)
    
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Phi_H_x=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y
    
    for i in range(n_hb):
     for j in range(n_hb):
       for m in range(n_hb):
          for q in range(n_hb):            
            k_x_i=2*np.pi*(i+1) # This goes with sines
            k_x_j=np.pi/2*(2*j+1) # This goes with cosines
            k_y_m=2*np.pi*(m+1) # This goes with sines
            k_y_q=np.pi/2*(2*q+1) # This goes with cosines
            # To take the differentiation, we use automatic diff style:
            sin_k_i_x=np.sin(k_x_i*XG); cos_k_i_x=np.cos(k_x_i*XG)
            cos_k_j_x=np.cos(k_x_j*XG); sin_k_j_x=np.sin(k_x_j*XG)
            sin_k_m_y=np.sin(k_y_m*YG); 
            cos_k_q_y=np.cos(k_y_q*YG); 
                                    
            # Assign the column of Phi_H
            Prime=-(k_x_j*sin_k_i_x*sin_k_j_x-k_x_i*cos_k_i_x*cos_k_j_x)   
            Phi_H_x[:,count]=Prime*sin_k_m_y*cos_k_q_y
            count+=1  

    return Phi_H_x

def Phi_H_2D_y(XG,YG,n_hb):
    """
    Create the derivatives along y, Phi_y, for the n_hb harmonic bases, 
    computed on the points (XG,YG)
    """
    # Get the number of points
    n_p=len(XG)
    
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Phi_H_y=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y
    
    for i in range(n_hb):
     for j in range(n_hb):
       for m in range(n_hb):
          for q in range(n_hb):            
            k_x_i=2*np.pi*(i+1) # This goes with sines
            k_x_j=np.pi/2*(2*j+1) # This goes with cosines
            k_y_m=2*np.pi*(m+1) # This goes with sines
            k_y_q=np.pi/2*(2*q+1) # This goes with cosines
            # To take the differentiation, we use automatic diff style:
            sin_k_i_x=np.sin(k_x_i*XG); 
            cos_k_j_x=np.cos(k_x_j*XG); 
            sin_k_m_y=np.sin(k_y_m*YG); cos_k_m_y=np.cos(k_y_m*YG)
            cos_k_q_y=np.cos(k_y_q*YG); sin_k_q_y=np.sin(k_y_q*YG)
                                    
            # Assign the column of Phi_H
            Prime=-(k_y_q*sin_k_m_y*sin_k_q_y-k_y_m*cos_k_m_y*cos_k_q_y)   
            Phi_H_y[:,count]=Prime*sin_k_i_x*cos_k_j_x
            count+=1  

    return Phi_H_y


def Phi_2_D_N_H(X_N,Y_N,n_hb,n_x,n_y):
    """
    Create the Phi_n operator for the RBF bases with collocation points (X_C,Y_C) and 
    shape factors c_k, computed on the points (X_N,Y_N) and 
    given the normal components (n_x,n_y). This is useful for defining Neuman conditions
    """
    Phi_N=Phi_H_2D_x(X_N,Y_N,n_hb)*n_x+Phi_H_2D_y(X_N,Y_N,n_hb)*n_y  
    
    return Phi_N




