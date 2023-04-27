# -*- coding: utf-8 -*-
"""
Latest update on Thu Apr 27 13:40:44 2023

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
 

    
class spicy:
    # 1. Initialize the class with the data
    def __init__(self, data, grid_point, basis='gauss', ST=None):
        """
        Initialization of an instance of the spicy class.
             
        # The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param model: string
            This defines the model. Currently, SPICY supports 4 models:
                1. 'scalar', to regress a scalar quantity.
                    This is implemented both in 2D and 3D.
                2. 'laminar', to regress the velocity field without turbulence modeling.
                    This is implemented both in 2D and 3D.
                3. 'RANSI', to regress a velocity field with a RANS model assuming isotropic Reynolds stresses (hence mean(u'**2)) is the only extra expected quantity.
                    This must be provided as the fourth entry of 'data', which becomes [u,v,w,uu].
                    NOTE: Currently, RANSI is only implemented in 3D.
                4. 'RANSA', to regress a velocity field with a RANS model without assuming isotropic Reynolds stresses.
                    this becomes [uu, vv, uv] in 2D and [uu, vv, ww, uv, uw, vw] in 3D.
                    NOTE: Currently, RANSA is only implemented in 3D.                            
        
        :param data: list
            If the instance is to be used to solve a regression problem, this 
            list contains the target data.
            This is an array [u] if the model is scalar,
            two arrays [u, v] for a 2D vector field and [u, v, w] for a 3D field.
            If the instance is to be used to solve the Poisson equation, this list contains
            the forcing term on the RHS of the Poisson equation.
                    
        :param grid_point: list
            Is a list of arrays containing the grid point [X_G ,Y_G] in 2D and [X_G, Y_G, Z_G] in 3D.   
                    
        :param basis: string
            This defines the basis. Currently, the two options are:
             1. 'gauss', i.e. Gaussian RBFs exp(-c_r**2*d(x))
             2. 'c4', i.e. C4 RBFs (1+d(x+)/c_r)**5(1-d(x+)/c_r)**5
        
        :param ST: list
            Is a list of arrays collecting Reynolds stresses. This is empty if
            the model is 'scalar' or 'laminar'.
            If the model is RANSI, it contains [uu']. 
            If the model is RANSA, it contains [uu, vv, uv] in 2D and 
              [uu, vv, ww, uv, uw, vw] in 3D.                                                   
                            
        ------------------------------------------------------------------------
        Other attributes
        ------------------------------------------------------------------------
        
        General attributes:
        
        X_G, Y_G, Z_G: coordinates of the point in which the data is available
        u : function to learn or u component in case of velocity field
        v: v component in case of velocity field (absent for scalar)
        w: w component in case of velocity field (absent for scalar)
        
        If constraints are assigned:
            
        X_D, Y_D, Z_D: coordinates of the points with D conditions
        c_D: values of the D conditions
        
        X_N,Y_N,Z_N: coordinates of the points with N conditions
        n_x,n_y,n_z: normal versors where N conditions are introduced
        c_N_X, c_N_Y, c_N_z: values of the N conditions
        
        X_Div, Y_Div, Z_Div: coordinates of the points with Div conditions
                
        If clustering is done:
            
        r_mM: vector collecting minimum (m) and maximum (M) radious of the RBFs 
        eps_l: scalar controlling the value of an RBF at the closest RBF neighbor               
        X_C,Y_C,Z_C : coordinates of the cluster centers/collocations 
        c_k: shape parameters of the RBFs 
        d_k: diameters of the rbfs  
       
        If problem is assembled:
        
        A: matrix A in the linear system
        B: matrix B in the linear system
        b_1: vector b_1 in the linear systems
        b_2: vector b_2 in the linear system       
        
        If computation is done: 
            
        weights: weights of the RBF regression 
        lambda: Lagrange multipliers of the RBF regression  
        
        RSI: Reynolds stress in case of isotropic flow (active for RANSI)
        
        [...] TODO: RANS formulations to be added.
        
        ------------------------------------------------------------------------
        Methods
        ------------------------------------------------------------------------
        
        """
        
        # Check the input is correct
        assert type(data) == list, 'Input data must be a list'
        assert type(grid_point) == list, 'Input grid_point must be a list'
        assert type(basis) == str, 'Basis must be a string'
        assert ST == None or type(ST) == list, 'ST must be a string or a list'
                        
        # Assign the basis
        if basis == 'gauss' or basis == 'c4':
            self.basis = basis
        else:
            raise ValueError('Wrong basis type, must be either \'gauss\' or \'c4\'')
        
        # Check the length of the grid points to see if it is 2D or 3D
        if len(grid_point)==2: # 2D problem
            self.type='2D'
            self.X_G=grid_point[0]
            self.Y_G=grid_point[1]
            # check the data 
            if len(data) == 1: # scalar
                self.model = 'scalar'
                self.u = data[0]
            elif len(data) == 2: # laminar
                self.model = 'laminar'
                self.u = data[0]
                self.v = data[1]
            else:
                raise ValueError('When grid_point is [X_g, Y_g], \'data\' must either be [u] or [u,v]')
            if ST is not None: # reynolds stress model
                raise ValueError('RANSI/RANSA currently not implemented in 2D')
            
        elif len(grid_point)==3: # 3D problem
            self.type='3D'
            self.X_G=grid_point[0]
            self.Y_G=grid_point[1]
            self.Z_G=grid_point[2] 
            if len(data) == 1:
                self.u=data[0]
                self.model = 'scalar'
            elif len(data) == 3:
                self.u=data[0]
                self.v=data[1]
                self.w=data[2]
                self.model = 'laminar'
            if ST is not None:
                raise ValueError('RANSI/RANSA currently not implemented in 3D')
        else:
            raise ValueError('Invalid size of input grid, currently only implemented in 2D and 3D')
            
        # Assign the number of data points. This is the same in all cases
        self.n_p = len(self.X_G)
        
                
        return
    
    
    # 2. Clustering (this does not depend on the model, but only on the dimension).
    def clustering(self, n_K, r_mM=[0.01,0.3], eps_l=0.7):
        """
        This function defines the collocation of a set of RBFs using the multi-level clustering
        described in the article. The function must be run before the constraint definition.
         
        The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param n_K: list
            This contains the n_k vector in eq 33. if n_K=[4,10], it means that the clustering
            will try to have a first level with RBFs whose size seeks to embrace 4 points, while the 
            second level seeks to embrace 10 points, etc.
            The length of this vector automatically defines the number of levels.
        
        :param r_mM: list, default = [0.01, 0.3]
            This contains the minimum and the maximum RBF's radiuses. This is defined as the distance from the
            collocation point at which the RBF value is 0.5.
                
        :param eps_l: float, default = 0.7
            This is the value that a RBF will have at its closest neighbour. It is used to define the shape
            factor from the clustering results.
                   
        """
        
        # Check the input is correct
        assert type(n_K) == list, 'Clustering levels must be given as a list'
        assert type(r_mM) == list and len(r_mM) == 2, 'r_mM must be a list of length 2'
        assert r_mM[0] < r_mM[1], 'Minimum radius must be smaller than maximum radius'
        assert eps_l < 1 and eps_l > 0, 'eps_l must be between zero and 1'
        
        # we assign the clustering parameters to self
        # they are needed in the constraints to set the shape parameters for the
        # RBFs which are located at constraint points
        
        self.r_mM = r_mM
        self.eps_l =eps_l
        
        # Check if we are dealing with a 2D or a 3D case
        if self.type=='2D': # This is 2D
            # Number of data points
            # Stack the coordinates in a matrix:
            Data_matrix = np.column_stack((self.X_G, self.Y_G))
            # Number of levels
            n_l = len(n_K)  
            
            # Loop over the number of levels
            for l in range(n_l):
                # Define number of clusters
                Clust = int(np.ceil(self.n_p / n_K[l])) 
                # Initialize the cluster function
                model = MiniBatchKMeans(n_clusters=Clust, random_state=0)    
                # Run the clustering and return the indices (optional)
                y_P = model.fit_predict(Data_matrix)
                # Obtaining the centers of the points
                Centers = model.cluster_centers_
                
                # Get the nearest neighbour of each center
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Centers)
                distances, indices = nbrs.kneighbors(Centers)
                sigma1 = distances[:,1]
                
                count = np.bincount(y_P, minlength = Clust) 
                sigma1[sigma1==0]=np.amax(sigma1[sigma1!=0]) 
                sigma1[count==1]=np.amax(sigma1) 
                
                # Pre-assign the collocation points
                X_C1 = Centers[:,0]
                Y_C1 = Centers[:,1]
                
                # Assign the results to a vector of collocation points
                if l == 0: # If this is the first layer, just assign:
                    X_C = X_C1 
                    Y_C = Y_C1 
                    sigma = sigma1 
                else: # Stack onto the existing ones
                    X_C = np.hstack((X_C, X_C1))
                    Y_C = np.hstack((Y_C, Y_C1))
                    sigma = np.hstack((sigma, sigma1))
                print('Clustering level '+str(l)+' completed')
            
            # Assign to the class
            self.X_C = X_C
            self.Y_C = Y_C
        
        elif self.type == '3D': # This is 3D
            # Stack the coordinates in a matrix:
            Data_matrix = np.column_stack((self.X_G, self.Y_G, self.Z_G))
            # Number of levels
            n_l = len(n_K)
            
            # Loop over the number of levels
            for l in range(n_l):
                # Define number of clusters
                Clust = int(np.ceil(self.n_p / n_K[l])) 
                # Initialize the cluster function
                model = MiniBatchKMeans(n_clusters=Clust, random_state=0)    
                # Run the clustering and return the indices (optional)
                y_P = model.fit_predict(Data_matrix)
                # Obtaining the centers of the points
                Centers = model.cluster_centers_
                
                # Get the nearest neighbour of each center
                nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Centers)
                distances, indices = nbrs.kneighbors(Centers)
                sigma1 = distances[:,1]
                
                count = np.bincount(y_P, minlength = Clust) 
                sigma1[sigma1==0]=np.amax(sigma1[sigma1!=0]) 
                sigma1[count==1]=np.amax(sigma1) 
                
                # Pre-assign the collocation points
                X_C1 = Centers[:,0]
                Y_C1 = Centers[:,1]
                Z_C1 = Centers[:,2]
                
                # Assign the results to a vector of collocation points
                if l == 0: # If this is the first layer, just assign:
                    X_C = X_C1 
                    Y_C = Y_C1 
                    Z_C = Z_C1 
                    sigma = sigma1 
                else: # Stack onto the existing ones
                    X_C = np.hstack((X_C, X_C1))
                    Y_C = np.hstack((Y_C, Y_C1))
                    Z_C = np.hstack((Z_C, Z_C1))
                    sigma = np.hstack((sigma, sigma1))
                print('Clustering level '+str(l)+' completed')

            # Assign to the class
            self.X_C = X_C
            self.Y_C = Y_C
            self.Z_C = Z_C
            
        # We conclude with the computation of the shape factors. These depend
        # on the type of RBF but not whether the type is 2D or 3D.
        if self.basis =='gauss':
            # Set the max and min values of c_k
            c_min = 1/(2*r_mM[1])*np.sqrt(np.log(2))
            c_max = 1/(2*r_mM[0])*np.sqrt(np.log(2))
            # compute the c_k 
            c_k = np.sqrt(-np.log(eps_l))/sigma
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # for plotting purposes, we store also the diameters
            d_k = 1/c_k*np.sqrt(np.log(2))
            
        elif self.basis == 'c4':
            # Set the max and min values of c_k
            c_min = 2*r_mM[0] / np.sqrt(1 - 0.5**0.2)
            c_max = 2*r_mM[1] / np.sqrt(1 - 0.5**0.2)
            # compute the c _k
            c_k = sigma / np.sqrt(1 - eps_l**0.2)
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # for plotting purposes, we store also the diameters
            d_k = c_k * np.sqrt(1 - 0.5**0.2)
        self.c_k = c_k
        self.d_k = d_k
        
        return
        
    # 3. Constraints.
    
    # We have two sorts of constraints: scalar and vector.
    # scalar apply to model = scalar and to the poisson solvers.
    # vector apply to all the other models.
    
    # the scalar ones include: Dirichlet and Neuman.
    # the vector one include: Dirichlet, Neuman and Div free.

    # 3.1 Scalar constraints
    def scalar_constraints(self, DIR=[], NEU=[], extra_RBF = True):
        """         
        This functions sets the boundary conditions for a scalar problem. The
        function must be run after the clustering was carried out.
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param DIR: list
            This contains the info for the Dirichlet conditions.
            If the model is 2D, then this has [X_D, Y_D, c_D].
            If the model is 3D, then this has [X_D, Y_D, Z_D, c_D].
              
        Here X_D, Y_D, Z_D are the coordinates of the poins where the value c_D
        is set.
        
        :param NEU: list
            This contains the info for the Neuman conditions.
            If the model is 2D, then this has [X_N, Y_N, n_x, n_y, c_N].
            If the model is 3D, then this has [X_N, Y_N, Z_n, n_x, n_y, n_z, c_N].
                   
        Here X_N, Y_N, Z_N are the coordinates of the poins where the value c_N
        is set for the directional derivative along the normal direction n_x, n_y, n_z
    
        :param extra_RBF: bool, default = True
            This is a flag to put extra collocation points where a constraint is
            set. It can improve the solution of the linear system as constraints
            remove degrees of freedom
                 
        NOTE: there is no check on the possible (dangerous) overlapping of conditions.
        Therefore, at the moment, one might put both Neuman and Dirichlet conditions
        at the same points. This is of course a terrible idea.
        TODO in future release: if both D and N conditions are given in at the same points
        (or to close?) then only one of them (e.g. D) is considered
        """
        
        # Check the input is correct
        assert type(DIR) == list, 'DIR must be a list'
        assert type(NEU) == list, 'NEU must be a list'
        assert type(extra_RBF) == bool, 'extra_RBF must be a boolean'
        
        # Check for Dirichlet conditions
        
        if not DIR: # We have no Dirichlet conditions
            # We still assign empty arrays so that the assembly of the system is easier
            self.n_D = 0
            self.X_D = np.array([])
            self.Y_D = np.array([])
            self.c_D = np.array([])
            # In 3D, we must add the z term
            if self.type == '3D':
                self.Z_D = np.array([])
                
        else: # We have Dirichlet conditions
            # Check if we have 2D or a 3D problem.
            if len(DIR)==3 and self.type == '2D': # 2D
                self.n_D=len(DIR[0])
                self.X_D=DIR[0]
                self.Y_D=DIR[1]
                self.c_D=DIR[2]
                
            elif len(DIR) == 4 and self.type == '3D': # 3D 
                self.n_D=len(DIR[0])
                self.X_D=DIR[0]
                self.Y_D=DIR[1]
                self.Z_D=DIR[2]
                self.c_D=DIR[3]
            else:
                raise ValueError('Length of Dirichlet conditions does not fit for Type ' + self.type)
  
        # Check for Neuman conditions
        
        if not NEU: # We have no Neumann conditions
            # We still assign empty arrays so that the assembly of the system is easier
            self.n_N = 0
            self.X_N = np.array([])
            self.Y_N = np.array([])
            self.c_N = np.array([])
            self.n_x = np.array([])
            self.n_y = np.array([])
            # In 3D, we must add the z term
            if self.type == '3D':
                self.Z_N = np.array([])
                self.n_z = np.array([])
                
        else: # We have Neumann conditions
            # Check if we have 2D or a 3D problem.
            if len(NEU) == 5 and self.type == '2D': # 2D
                self.n_N=len(NEU[0])
                self.X_N=NEU[0]
                self.Y_N=NEU[1]
                self.n_x=NEU[2]
                self.n_y=NEU[3]
                self.c_N=NEU[4]
                
            elif len(NEU) == 7 and self.type == '3D': # 3D
                self.n_N=len(NEU[0])
                self.X_N=NEU[0]
                self.Y_N=NEU[1]
                self.Z_N=NEU[2]            
                self.n_x=NEU[3]
                self.n_y=NEU[4]
                self.n_z=NEU[5]            
                self.c_N=NEU[6] 
                
            else:
                raise ValueError('Length of Neumann conditions does not fit for Type ' + self.type)
        
        # if self.type == '2D':
        #     # For this, we first get the points where a constraint is set
        #     X_constr = np.concatenate((self.X_D, self.X_N))
        #     Y_constr = np.concatenate((self.Y_D, self.Y_N))
        #     _, valid_idcs = np.unique(np.column_stack((X_constr, Y_constr)), return_index = True, axis = 0)
        #     # Extract the unique coordinate sets
        #     X_unique = X_constr[valid_idcs]
        #     Y_unique = Y_constr[valid_idcs]
            
        #     # Here, we stack three different coordinate sets. These are the unique 
        #     # constraint points, the Dirichlet, Divergance and the Neumann constraints
        #     points_dir_neu = np.column_stack((X_unique, Y_unique))
        #     points_dir = np.column_stack((self.X_D, self.Y_D))
        #     points_neu = np.column_stack((self.X_N, self.Y_N))
            
        #     # this creates a boolean array indicating whether a given point in 
        #     # (X_unqiue, Y_unique) is contained in the Dirichlet conditions
        #     shared_points_dir_neu = (points_dir_neu[:, None] == points_dir).all(-1).any(1)
        #     # Note: We do not need to extract the Dirichlet conditions, because
        #     # we always keep them. Thus, they are not reassigned. However, the
        #     # remaining Dirichlet and Neumann points must be extracted
        #     X_N_reduced = X_unique[~shared_points_dir_neu]
        #     Y_N_reduced = Y_unique[~shared_points_dir_neu]
            
        #     # We stack them again as a coordinate set
        #     points_neu_out = np.column_stack((X_N_reduced, Y_N_reduced))
        #     # This boolean array indicates which of the original Neumann constraints
        #     # is still active. This is needed to also extract the relevant values
        #     # of the constraints. For divergence-free, this is not needed as they 
        #     # are always 0
        #     share_points_neumann_before_and_after_reduction = (points_neu_out == points_neu[:, None]).all(-1).any(1)
        #     # And we extract the final values
        #     self.X_N = self.X_N[share_points_neumann_before_and_after_reduction]
        #     self.Y_N = self.Y_N[share_points_neumann_before_and_after_reduction]
        #     self.n_x = self.n_x[share_points_neumann_before_and_after_reduction]
        #     self.n_y = self.n_y[share_points_neumann_before_and_after_reduction]
        #     self.c_N = self.c_N[share_points_neumann_before_and_after_reduction]
        #     self.n_N = self.X_N.shape[0]
        
        # Finally, we add the extra RBFs in the constraint points if desired
        if extra_RBF == True:
            # Check if we have 2D or a 3D problem.
            if self.type == '2D': # 2D
                # Assemble all the points where we have constraints
                X_constr = np.concatenate((self.X_N, self.X_D))
                Y_constr = np.concatenate((self.Y_N, self.Y_D))
                # Get the unique values 
                unique_values = np.unique(np.column_stack((X_constr, Y_constr)), axis = 0)
                X_unique = unique_values[:,0]
                Y_unique = unique_values[:,1]
                # Get the additional RBF shape parameters
                c_k, d_k = add_constraint_collocations_2D(X_unique, Y_unique,
                    self.X_C, self.Y_C, self.r_mM, self.eps_l, self.basis)
                # Concatenate them with the existing collocation points
                self.c_k = np.concatenate((self.c_k, c_k))
                self.d_k = np.concatenate((self.d_k, d_k))
                self.X_C = np.concatenate((self.X_C, X_unique))
                self.Y_C = np.concatenate((self.Y_C, Y_unique))      
            
            elif self.type == '3D': # 3D
                # Assemble all the points where we have constraints
                X_constr = np.concatenate((self.X_N, self.X_D))
                Y_constr = np.concatenate((self.Y_N, self.Y_D))
                Z_constr = np.concatenate((self.Z_N, self.Z_D))
                # Get the unique values 
                unique_values = np.unique(np.column_stack((X_constr, Y_constr, Z_constr)), axis = 0)
                X_unique = unique_values[:,0]
                Y_unique = unique_values[:,1]
                Z_unique = unique_values[:,2]
                # Get the additional RBF shape parameters
                c_k, d_k = add_constraint_collocations_3D(X_unique, Y_unique, Z_unique,
                    self.X_C, self.Y_C, self.Z_C, self.r_mM, self.eps_l, self.basis)
                # Concatenate them with the existing collocation points
                self.c_k = np.concatenate((self.c_k, c_k))
                self.d_k = np.concatenate((self.d_k, d_k))
                self.X_C = np.concatenate((self.X_C, X_unique))
                self.Y_C = np.concatenate((self.Y_C, Y_unique))  
                self.Z_C = np.concatenate((self.Z_C, Z_unique))          
        
        # Summary output for the user
        print(str(self.n_D)+' Dirichlet conditions assigned') 
        print(str(self.n_N)+' Neumann conditions assigned')
        
        return


    # 3.2 Scalar constraints
    def vector_constraints(self, DIR=[], NEU=[], DIV=[], extra_RBF = True):
        """        
        # This functions sets the boundary conditions for a laminar problem. The
        function must be run after the clustering was carried out.
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param DIR: list
            This contains the info for the Dirichlet conditions.
            If the model is 2D, then this has [X_D, Y_D, c_D_X, c_D_Y].
            If the model is 3D, then this has [X_D, Y_D, Z_D, c_D_X, c_D_Y, c_D_Z].
              
            Here X_D, Y_D, Z_D are the coordinates of the poins where the value c_D_X,
            c_D_Y, c_D_Z is set in 2 or 3 dimensions.
        
        :param NEU: list
            This contains the info for the Neuman conditions.
            If the model is 2D, then this has [X_N, Y_N, n_x, n_y, c_N_X, c_N_Y].
            If the model is 3D, then this has [X_N, Y_N, Z_n, n_x, n_y, n_z, c_N_X, c_N_Y, c_N_Z].
                   
            Here X_N, Y_N, Z_N are the coordinates of the poins where the value c_N_X,
            c_N_Y, c_N_Z is set for the directional derivative along the 
            normal direction n_x,n_y,n_z
            
        :param DIV: list
            This contains the info for the Divergence free conditions.
            If the model is 2D, then this has [X_Div, Y_Div].
            If the model is 3D, then this has [X_Div, Y_Div, Z_Div].
        
        :param extra_RBF: bool, default = True
            This is a flag to put extra collocation points where a constraint is
            set. It can improve the solution of the linear system as constraints
            remove degrees of freedom
        
        NOTE: there is no check on the possible (dangerous) overlapping of conditions.
        Therefore, at the moment, one might put both Neuman and Dirichlet conditions
        at the same points. This is of course a terrible idea.
        TODO in future release: if both D and N conditions are given in at the same points
        ( or to close?) then only one of them (e.g. D) is considered
                
        """
        
        # Check the input is correct
        assert type(DIR) == list, 'DIR must be a list'
        assert type(NEU) == list, 'NEU must be a list'
        assert type(DIV) == list, 'DIV must be a list'
        assert type(extra_RBF) == bool, 'extra_RBF must be a boolean'
        
        # Check for Dirichlet conditions
        
        if not DIR: # We have no Dirichlet conditions
            # We still assign empty arrays so that the assembly of the system is easier
            self.n_D = 0
            self.X_D = np.array([])
            self.Y_D = np.array([])
            self.c_D_X = np.array([])
            self.c_D_Y = np.array([])
            # In 3D, we must add the z terms
            if self.type == '3D':
                self.Z_D = np.array([])
                self.c_D_Z = np.array([])
                
        else: # We have Dirichlet conditions
            # Check if we have 2D or a 3D problem
            if len(DIR) == 4 and self.type == '2D': # 2D
                self.n_D = len(DIR[0])
                self.X_D = DIR[0]
                self.Y_D = DIR[1]
                self.c_D_X = DIR[2]
                self.c_D_Y = DIR[3]
                       
            elif len(DIR) == 6 and self.type == '3D': # 3D
                self.n_D = len(DIR[0])
                self.X_D = DIR[0]
                self.Y_D = DIR[1]
                self.Z_D = DIR[2]
                self.c_D_X = DIR[3]
                self.c_D_Y = DIR[4]
                self.c_D_Z = DIR[5]
                
            else:
                raise ValueError('Length of Dirichlet conditions is wrong for type \'' + self.type + '\'')
           
        # Check for Neumann conditions
        
        if not NEU: # We have no Neumann conditions
            # We still assign empty arrays so that the assembly of the system is easier
            self.n_N = 0
            self.X_N = np.array([])
            self.Y_N = np.array([])
            self.c_N_X = np.array([])
            self.c_N_Y = np.array([])
            self.n_y = np.array([])
            self.n_x = np.array([])
            # In 3D, we must add the z terms
            if self.type == '3D':
                self.Z_N = np.array([])
                self.c_N_Z = np.array([])
                self.n_z = np.array([])
                
        else:  # We have Neumann conditions
            # Check if we have 2D or a 3D problem
            if len(NEU) == 6 and self.type == '2D':  # 2D
                self.n_N = len(NEU[0])
                self.X_N = NEU[0]
                self.Y_N = NEU[1]
                self.n_x = NEU[2]
                self.n_y = NEU[3]
                self.c_N_X = NEU[4]
                self.c_N_Y = NEU[5]
                
            elif len(NEU) == 9 and self.type == '3D': # 3D
                self.n_N = len(NEU[0])
                self.X_N = NEU[0]
                self.Y_N = NEU[1]
                self.Z_N = NEU[2]            
                self.n_x = NEU[3]
                self.n_y = NEU[4]
                self.n_z = NEU[5]            
                self.c_N_X = NEU[6]          
                self.c_N_Y = NEU[7]          
                self.c_N_Z = NEU[8] 
                
            else:
                raise ValueError('Length of Neumann conditions is wrong for type \'' + self.type + '\'')
                
        # Check for Divergence conditions
        # TODO This check should be obsolete as the vector constraints only make sense 
        # when we couple the regression with a divergence free constraint. Instead,
        # a warning/error should be thrown        
        if not DIV:
            # We still assign empty arrays so that the assembly of the system is easier
            self.n_Div = 0
            self.X_Div = []
            self.Y_Div = []
            # In 3D, we must add the z terms
            if self.type == '3D':
                self.Z_Div = np.array([])
        else:
            #Check if we have 2D or a 3D problem
            if len(DIV) == 2: # this means 2D
                self.n_Div = len(DIV[0])
                self.X_Div = DIV[0]
                self.Y_Div = DIV[1]
                
            else:
                self.n_Div = len(DIV[0])
                self.X_Div = DIV[0]
                self.Y_Div = DIV[1]
                self.Z_Div = DIV[2]
                
        # if self.type == '2D':
        #     # For this, we first get the points where a constraint is set
        #     X_constr = np.concatenate((self.X_D, self.X_N, self.X_Div))
        #     Y_constr = np.concatenate((self.Y_D, self.Y_N, self.Y_Div))
        #     _, valid_idcs = np.unique(np.column_stack((X_constr, Y_constr)), return_index = True, axis = 0)
        #     # Extract the unique coordinate sets
        #     X_unique = X_constr[valid_idcs]
        #     Y_unique = Y_constr[valid_idcs]
            
        #     # Here, we stack three different coordinate sets. These are the unique 
        #     # constraint points, the Dirichlet, Divergance and the Neumann constraints
        #     points_dir_div_neu = np.column_stack((X_unique, Y_unique))
        #     points_dir = np.column_stack((self.X_D, self.Y_D))
        #     points_neu = np.column_stack((self.X_N, self.Y_N))
        #     points_div = np.column_stack((self.X_Div, self.Y_Div))
            
        #     # this creates a boolean array indicating whether a given point in 
        #     # (X_unqiue, Y_unique) is contained in the Dirichlet conditions
        #     shared_points_dir_div_neu = (points_dir_div_neu[:, None] == points_dir).all(-1).any(1)
        #     # Note: We do not need to extract the Dirichlet conditions, because
        #     # we always keep them. Thus, they are not reassigned. However, the
        #     # remaining Dirichlet and Neumann points must be extracted
        #     X_Div_Neu = X_unique[~shared_points_dir_div_neu]
        #     Y_Div_Neu = Y_unique[~shared_points_dir_div_neu]
            
        #     # We stack them again as a coordinate set
        #     points_div_neu = np.column_stack((X_Div_Neu, Y_Div_Neu))
        #     # And extract a boolena array indicating whether the points in that set
        #     # are also contained in the divergence-free set
        #     shared_points_div_neu = (points_div_neu[:, None] == points_div).all(-1).any(1)
        #     # We extract and assign the remaining divergence-free constraints
        #     self.X_Div = X_Div_Neu[shared_points_div_neu]
        #     self.Y_Div = Y_Div_Neu[shared_points_div_neu]
        #     self.n_Div = self.X_Div.shape[0]
        #     # as well as the remaining Neumann conditions. This the set of points
        #     # which are not contained in either the Dirichlet or divergence-free points
        #     X_N_reduced = X_Div_Neu[~shared_points_div_neu]
        #     Y_N_reduced = Y_Div_Neu[~shared_points_div_neu]
            
        #     # We stack them again as a coordinate set
        #     points_neu_out = np.column_stack((X_N_reduced, Y_N_reduced))
        #     # This boolean array indicates which of the original Neumann constraints
        #     # is still active. This is needed to also extract the relevant values
        #     # of the constraints. For divergence-free, this is not needed as they 
        #     # are always 0
        #     share_points_neumann_before_and_after_reduction = (points_neu_out == points_neu[:, None]).all(-1).any(1)
        #     # And we extract the final values
        #     self.X_N = self.X_N[share_points_neumann_before_and_after_reduction]
        #     self.Y_N = self.Y_N[share_points_neumann_before_and_after_reduction]
        #     self.n_x = self.n_x[share_points_neumann_before_and_after_reduction]
        #     self.n_y = self.n_y[share_points_neumann_before_and_after_reduction]
        #     self.c_N_X = self.c_N_X[share_points_neumann_before_and_after_reduction]
        #     self.c_N_Y = self.c_N_Y[share_points_neumann_before_and_after_reduction]
        #     self.n_N = self.X_N.shape[0]
        
        # Finally, we add the extra RBFs in the constraint points if desired
        if extra_RBF == True:
            # Check if we have 2D or a 3D problem.
            if self.type == '2D': # 2D
                # Assemble all the points where we have constraints
                X_constr = np.concatenate((self.X_D, self.X_N, self.X_Div))
                Y_constr = np.concatenate((self.Y_D, self.Y_N, self.Y_Div))
                # Get the unique values 
                unique_values = np.unique(np.column_stack((X_constr, Y_constr)), axis = 0)
                X_unique = unique_values[:,0]
                Y_unique = unique_values[:,1]
                # Get the additional RBF shape parameters
                c_k, d_k = add_constraint_collocations_2D(X_unique, Y_unique,
                    self.X_C, self.Y_C, self.r_mM, self.eps_l, self.basis)
                # Concatenate them with the existing collocation points
                self.c_k = np.concatenate((self.c_k, c_k))
                self.d_k = np.concatenate((self.d_k, d_k))
                self.X_C = np.concatenate((self.X_C, X_unique))
                self.Y_C = np.concatenate((self.Y_C, Y_unique))      
            
            elif self.type == '3D': # 3D
                # Assemble all the points where we have constraints
                X_constr = np.concatenate((self.X_D, self.X_N, self.X_Div))
                Y_constr = np.concatenate((self.Y_D, self.Y_N, self.Y_Div))
                Z_constr = np.concatenate((self.Z_D, self.Z_N, self.Z_Div))
                # Get the unique values 
                unique_values = np.unique(np.column_stack((X_constr, Y_constr, Z_constr)), axis = 0)
                X_unique = unique_values[:,0]
                Y_unique = unique_values[:,1]
                Z_unique = unique_values[:,2]
                # Get the additional RBF shape parameters
                c_k, d_k = add_constraint_collocations_3D(X_unique, Y_unique, Z_unique,
                    self.X_C, self.Y_C, self.Z_C, self.r_mM, self.eps_l, self.basis)
                # Concatenate them with the existing collocation points
                self.c_k = np.concatenate((self.c_k, c_k))
                self.d_k = np.concatenate((self.d_k, d_k))
                self.X_C = np.concatenate((self.X_C, X_unique))
                self.Y_C = np.concatenate((self.Y_C, Y_unique))  
                self.Z_C = np.concatenate((self.Z_C, Z_unique))  
        
        # Summary output for the user
        print(str(self.n_D)+' D conditions assigned') 
        print(str(self.n_N)+' N conditions assigned')
        print(str(self.n_Div)+' Div conditions assigned')
        
        return
        
    # 3.3 Plot the RBFs, this is just a visualization tool
    def plot_RBFs(self):
        """
        Utility function to check the spreading of the RBFs after the clustering.
        No input is required, nothing is assigned to SPICY and no output is generated.
        """
        
        # check if it is 2D or 3D
        if self.type == '2D': # 2D
            try:  
                fig, axs = plt.subplots(1, 2, figsize = (10, 5), dpi = 100)
                # First plot is the RBF distribution
                axs[0].set_title("RBF Collocation")
                for i in range(0,len(self.X_C),1):
                    circle1 = plt.Circle((self.X_C[i], self.Y_C[i]), self.d_k[i]/2, 
                                          fill=True,color='g',edgecolor='k',alpha=0.2)
                    axs[0].add_artist(circle1)  
                # also show the data points
                # if self.model == 'scalar':
                #     axs[0].scatter(self.X_G, self.Y_G, c=self.u, s=10)
                # elif self.model == 'laminar':
                #     axs[0].scatter(self.X_G, self.Y_G, c=np.sqrt(self.u**2 + self.v**2), s=10)
                
                # also show the constraints if they are set
                axs[0].plot(self.X_D, self.Y_D,'ro')
                axs[0].plot(self.X_N, self.Y_N,'bs')
                # if self.model == 'laminar':
                #     axs[0].plot(self.X_Div, self.Y_Div, 'gd')
               # axs[0].set_xlim(-0.52, 0.52) # let is automatic; we do not know the domain
               # axs[0].set_ylim(-0.52, 0.52)
 
                # second plot is the distribution of diameters:
                axs[1].stem(self.d_k)
                axs[1].set_xlabel('Basis index')
                axs[1].set_ylabel('Diameter') 
                axs[1].set_title("Distribution of diameters")
                fig.tight_layout()
           
            except:
                raise ValueError('Problems in plotting. Set constraints and cluster!')   
       
        elif self.type == '3D': # 3D
            try:
                # for now, we just show the distribution of diameters, as 3D sphere
                # visualizations are very difficult
                fig, ax = plt.subplots(figsize = (5, 5), dpi = 100)
                ax.set_title("RBF Collocation")
                ax.stem(self.d_k)
                ax.set_xlabel('Basis index')
                ax.set_ylabel('Diameter') 
                ax.set_title("Distribution of diameters")
                fig.tight_layout()
            except:
                raise ValueError('Problems in plotting. Set constraints and cluster!')  
        return


    # 4. Assembly A, B, b_1, b_2
    
    # We have two sorts of assemblies: poisson and regression.
    # poisson applies to the poisson solvers.
    # regression applies to scalar and laminar regression.
    
    # the poisson one includes the source terms on the r.h.s..
    # the regression one inlcudes a potential penalty of a divergence free flow.

    # 4.1. Poisson solver
    def Assembly_Poisson(self, n_hb=0):
        """
        This function assembly the matrices A, B, C, D from the paper.
        TODO. Currently implemented only for model='scalar'
        
        #The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------

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
        
        """   
        
        # assert type(source_terms) == np.ndarray, 'Source terms must be a 1D numpy array'
        assert type(n_hb) == int, 'Number of harmonic basis must be an integer'
        # assert len(source_terms.shape) == 1, 'Source terms must be a 1D numpy array'
        
        
        source_terms=self.u
        
        
        # Assign the number of harmonic basis functions
        self.n_hb = n_hb
        # get the number of basis and points as we need them a couple of times
        # 2D and 3D have different bases
        if self.type == '2D':
            self.n_b = self.X_C.shape[0] + n_hb**4
        elif self.type == '3D':
            self.n_b = self.X_C.shape[0] + n_hb**6
            
        if self.model=='scalar': 
            if self.type == '2D': # 2D           
                # get the rescaling factor by normalizing the r.h.s. of the source terms
                # TODO This should maybe also consider the B.C.?
                R=max(np.max(source_terms), -np.max(-source_terms))
                if R == 0:
                 self.rescale = 1    
                else:
                 self.rescale=R 
                                      
                # Approach 1: we build A, B, b1, b2 as in the article from Sperotto
                L = np.hstack((
                    Phi_H_2D_Laplacian(self.X_G, self.Y_G, self.n_hb),
                    Phi_RBF_2D_Laplacian(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
                    )) 
                # Then A and b1 are :
                self.A = 2*L.T@L
                self.b_1 = 2*L.T.dot(source_terms)/self.rescale
    
                # Check for Dirichlet
                if self.n_D != 0:
                    # Compute Phi on X_D
                    Matrix_D = np.hstack((
                        Phi_H_2D(self.X_D, self.Y_D, self.n_hb),
                        Phi_RBF_2D(self.X_D, self.Y_D, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))
                else: # No Dirichlet conditions
                    # initialize the empty array
                    Matrix_D = np.empty((0, self.n_b))
                
                # Check for Neumann
                if self.n_N != 0: # We have Neumann conditions
                    # Compute Phi_x on X_N
                    Matrix_Phi_2D_X_N_der_x = np.hstack((
                        Phi_H_2D_x(self.X_N, self.Y_N, self.n_hb),
                        Phi_RBF_2D_x(self.X_N, self.Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_N
                    Matrix_Phi_2D_X_N_der_y = np.hstack((
                        Phi_H_2D_y(self.X_N, self.Y_N, self.n_hb),
                        Phi_RBF_2D_y(self.X_N, self.Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))
                    # Compute Phi_n on X_N
                    Matrix_D_N = Matrix_Phi_2D_X_N_der_x*self.n_x[:, np.newaxis] +\
                        Matrix_Phi_2D_X_N_der_y*self.n_y[:, np.newaxis]
                else: # No Neumann conditions
                    # initialize the empty array
                    Matrix_D_N = np.empty((0, self.n_b))
                    
                # assemble B and b_2
                self.B = np.vstack((Matrix_D, Matrix_D_N)).T
                self.b_2 = np.concatenate((self.c_D, self.c_N))/self.rescale
                
            elif self.type == '3D': # 3D
                # get the rescaling factor by normalizing the r.h.s. of the source terms
                # TODO This should maybe also consider the B.C.?
                source_terms = self.u
                self.rescale = max(np.max(source_terms), -np.max(-source_terms))   
                if np.abs(self.rescale) < 1e-10:
                    self.rescale = 1
                
                # Approach 1: we build A, B, b1, b2 as in the article from Sperotto
                L = np.hstack((
                    Phi_H_3D_Laplacian(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                    Phi_RBF_3D_Laplacian(self.X_G, self.Y_G, self.Z_G,
                                         self.X_C, self.Y_C, self.Z_C,
                                         self.c_k, self.basis)
                    )) 
                # Then A and b1 are :
                self.A = 2*L.T@L
                self.b_1 = 2*L.T.dot(source_terms) / self.rescale
                
                # Check for Dirichlet
                if self.n_D != 0:
                    # Compute Phi on X_D
                    Matrix_D = np.hstack((
                        Phi_H_3D(self.X_D, self.Y_D, self.Z_D, self.n_hb),
                        Phi_RBF_3D(self.X_D, self.Y_D, self.Z_D,
                                   self.X_C, self.Y_C, self.Z_C,
                                   self.c_k, self.basis)
                        ))
                else: # No Dirichlet conditions
                    # initialize the empty array
                    Matrix_D = np.empty(0, self.n_b)
                    
                # Check for Neumann
                if self.n_N != 0: # We have Neumann conditions
                    # Compute Phi_x on X_N
                    Matrix_Phi_3D_X_N_der_x = np.hstack((
                        Phi_H_3D_x(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_x(self.X_N, self.Y_N, self.Z_N,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_N
                    Matrix_Phi_3D_X_N_der_y = np.hstack((
                        Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_z on X_N
                    Matrix_Phi_3D_X_N_der_z = np.hstack((
                        Phi_H_3D_z(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_z(self.X_N, self.Y_N, self.Z_N,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_n on X_N
                    Matrix_D_N = Matrix_Phi_3D_X_N_der_x*self.n_x[:, np.newaxis] +\
                                 Matrix_Phi_3D_X_N_der_y*self.n_y[:, np.newaxis] +\
                                 Matrix_Phi_3D_X_N_der_z*self.n_z[:, np.newaxis]
                else: # No Neumann conditions
                    # initialize the empty array
                    Matrix_D_N = np.empty(0, self.n_b)
                    
                # assemble B and b_2
                self.B = np.vstack((Matrix_D, Matrix_D_N)).T
                self.b_2 = np.concatenate((self.c_D, self.c_N))/self.rescale
        else:
            raise NotImplementedError('Assembly_Poisson only build for the scalar function') 
    
        return
    
    
    # 4.2. Regression
    def Assembly_Regression(self, n_hb=0, alpha_div=None):
        """
        This function assembly the matrices A, B, C, D from the paper.
        
        The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param source_terms: array
            This is relevant only in the 'scalar' model. 
            This vector contains the values for the source term on all the given points (term s in eq 27).
            To solve the Laplace equation, it should be a vector of zeros.
            To solve the Poisson equation for the pressure, this is the RHS of eq.21.
            In any case, the specification of the RHS is done outside the assmebly function.
             
        :param n_hb: int (currently  not recommended) 
            Also for a regression, the harmonic basis can improve the regression
            as they can model global trends which are similar to a low order
            polynomial. Furthermore, for homogenous problem, they automatically
            fulfill the boundary conditions.
            
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
           
        :param alpha_div: float (default: None)
            This enables a divergence free penalty in the entire flow field.
            Increasing this parameter penalizes errors in the divergence free 
            flow more. This is particularly important to obtain good derivatives 
            for the pressure computation
         """   
        # Assign the number of harmonic basis functions
        self.n_hb = n_hb
        # get the number of basis and points as we need them a couple of times
        # 2D and 3D have different bases
        if self.type == '2D':
            self.n_b = self.X_C.shape[0] + n_hb**4
        elif self.type == '3D':
            self.n_b = self.X_C.shape[0] + n_hb**6
        
        # Scalar model
        if self.model == 'scalar':
            if self.type == '2D': # 2D
                # define the rescaling factor which is done based on the maximum
                # absolute velocity that is available in u and v 
                self.rescale = self.u[np.argmax(np.abs(self.u))]
                
                # Check for Dirichlet
                if self.n_D !=0: # We have Dirichlet conditions
                    # Compute Phi on X_D (16)
                    Matrix_D = np.hstack((
                        Phi_H_2D(self.X_D, self.Y_D, self.n_hb),
                        Phi_RBF_2D(self.X_D, self.Y_D, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))
                else: # No Dirichlet conditions
                    # initialize the empty array
                    Matrix_D = np.empty((0, self.n_b))
                
                # Check for Neumann
                if self.n_N != 0: # We have Neumann conditions
                    # Compute Phi_x on X_N
                    Matrix_Phi_2D_X_N_der_x = np.hstack((
                        Phi_H_2D_x(self.X_N, self.Y_N, self.n_hb),
                        Phi_RBF_2D_x(self.X_N, self.Y_N,
                                      self.X_C, self.Y_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_N
                    Matrix_Phi_2D_X_N_der_y = np.hstack((
                        Phi_H_2D_y(self.X_N, self.Y_N, self.n_hb),
                        Phi_RBF_2D_y(self.X_N, self.Y_N,
                                      self.X_C, self.Y_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_n on X_N (equation (18))
                    Matrix_D_N = Matrix_Phi_2D_X_N_der_x*self.n_x[:, np.newaxis] +\
                                 Matrix_Phi_2D_X_N_der_y*self.n_y[:, np.newaxis]
                else: # No Neumann conditions
                    # initialize the empty array
                    Matrix_D_N = np.empty((0,self.n_b))
                
                # We can now assemble the matrix independent of what combinations
                # of Dirichlet and Neumann we have
                self.B = np.vstack((Matrix_D, Matrix_D_N)).T
                # We do the same for b_2, as this can also be done for every case
                self.b_2 = np.concatenate((self.c_D, self.c_N)) / self.rescale
                
                # We compute Phi on all node points X
                Matrix_Phi_2D_X = np.hstack((
                    Phi_H_2D(self.X_G, self.Y_G, self.n_hb),
                    Phi_RBF_2D(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # block structure of A as in equation(10)
                PhiT_dot_Phi = Matrix_Phi_2D_X.T.dot(Matrix_Phi_2D_X)
                self.A = 2*Matrix_Phi_2D_X.T.dot(Matrix_Phi_2D_X)
                # compute b_1
                self.b_1 = 2*Matrix_Phi_2D_X.T.dot(self.u) / self.rescale
                
            elif self.type == '3D':   
                # define the rescaling factor which is done based on the maximum
                # absolute velocity that is available in u and v 
                self.rescale = self.u[np.argmax(np.abs(self.u))]
                
                # Check for Dirichlet
                if self.n_D !=0: # We have Dirichlet conditions
                    # Compute Phi on X_D (16)
                    Matrix_D = np.hstack((
                        Phi_H_3D(self.X_D, self.Y_D, self.Z_D, self.n_hb),
                        Phi_RBF_3D(self.X_D, self.Y_D, self.Z_D,
                                   self.X_C, self.Y_C, self.Z_C,
                                   self.c_k, self.basis)
                        ))
                else: # No Dirichlet conditions
                    # initialize the empty array
                    Matrix_D = np.empty((0, self.n_b))
                
                # Check for Neumann
                if self.n_N != 0: # We have Neumann conditions
                    # Compute Phi_x on X_N
                    Matrix_Phi_3D_X_N_der_x = np.hstack((
                        Phi_H_3D_x(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_x(self.X_N, self.Y_N, self.Z_N,
                                      self.X_C, self.Y_C, self.Z_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_N
                    Matrix_Phi_3D_X_N_der_y = np.hstack((
                        Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
                                      self.X_C, self.Y_C, self.Z_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_z on X_N
                    Matrix_Phi_3D_X_N_der_z = np.hstack((
                        Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
                                      self.X_C, self.Y_C, self.Z_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_n on X_N (equation (18))
                    Matrix_D_N = Matrix_Phi_3D_X_N_der_x*self.n_x[:, np.newaxis] +\
                                 Matrix_Phi_3D_X_N_der_y*self.n_y[:, np.newaxis] +\
                                 Matrix_Phi_3D_X_N_der_z*self.n_z[:, np.newaxis]
                else: # No Neumann conditions
                    # initialize the empty array
                    Matrix_D_N = np.empty((0,self.n_b))
                
                # We can now assemble the matrix independent of what combinations
                # of Dirichlet and Neumann we have
                self.B = np.vstack((Matrix_D, Matrix_D_N)).T
                # We do the same for b_2, as this can also be done for every case
                self.b_2 = np.concatenate((self.c_D, self.c_N)) / self.rescale
                
                # We compute Phi on all node points X
                Matrix_Phi_3D_X = np.hstack((
                    Phi_H_3D(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                    Phi_RBF_3D(self.X_G, self.Y_G, self.Z_G, 
                               self.X_C, self.Y_C, self.Z_C,
                               self.c_k, self.basis)
                    ))
                # block structure of A as in equation(10)
                PhiT_dot_Phi = Matrix_Phi_3D_X.T.dot(Matrix_Phi_2D_X)
                self.A = 2*Matrix_Phi_3D_X.T.dot(Matrix_Phi_2D_X)
                # compute b_1
                self.b_1 = 2*Matrix_Phi_3D_X.T.dot(self.u) / self.rescale
                
        # Laminar model
        elif self.model == 'laminar':  
            # we need to check whether we are 2D or 3D laminar as this changes the assignment
            if self.type == '2D': # 2D
                # define the rescaling factor which is done based on the maximum
                # absolute velocity that is available in u and v
                data = np.concatenate((self.u, self.v))
                self.rescale = data[np.argmax(np.abs(data))]
                
                # Here, we compute the matrix D_nabla as in equation (15)
                # There is no check whether X_Div as it is assumed that the laminar
                # case is only used to enforce divergence free constraints
                # TODO implement a check that gives an error when laminar is called
                # and no divergence free constraints are used
                
                # compute the derivatives in x
                Matrix_Phi_2D_X_nabla_der_x = np.hstack((
                    Phi_H_2D_x(self.X_Div, self.Y_Div, self.n_hb),
                    Phi_RBF_2D_x(self.X_Div, self.Y_Div, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # compute the derivatives in y
                Matrix_Phi_2D_X_nabla_der_y = np.hstack((
                    Phi_H_2D_y(self.X_Div, self.Y_Div, self.n_hb),
                    Phi_RBF_2D_y(self.X_Div, self.Y_Div, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # stack them together to obtain X_nabla
                Matrix_D_nabla = np.hstack((Matrix_Phi_2D_X_nabla_der_x, Matrix_Phi_2D_X_nabla_der_y)) 
                
                # For the Neumann and Dirichlet B.C., we do the check whether 
                # we have them or not. If we have them, the matrices are computed
                # If not, we initialize an empty array of appropriate size. This
                # allows us to stack them together universally without checking all
                # four possibilities of having or not having the two
                
                # Check for Dirichlet
                if self.n_D !=0: # We have Dirichlet conditions
                    # Compute Phi on X_D (16)
                    Matrix_Phi_2D_D = np.hstack((
                        Phi_H_2D(self.X_D, self.Y_D, self.n_hb),
                        Phi_RBF_2D(self.X_D, self.Y_D, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))
                else: # No Dirichlet conditions
                    # initialize the empty array
                    Matrix_Phi_2D_D = np.empty((0, self.n_b))
                # stack into the block structure of equation (16)
                Matrix_D = np.block([
                    [Matrix_Phi_2D_D,np.zeros((self.n_D, self.n_b))],
                    [np.zeros((self.n_D, self.n_b)), Matrix_Phi_2D_D]
                    ])
                
                # Check for Neumann
                if self.n_N != 0: # We have Neumann conditions
                    # Compute Phi_x on X_N
                    Matrix_Phi_2D_X_N_der_x = np.hstack((
                        Phi_H_2D_x(self.X_N, self.Y_N, self.n_hb),
                        Phi_RBF_2D_x(self.X_N, self.Y_N,
                                      self.X_C, self.Y_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_N
                    Matrix_Phi_2D_X_N_der_y = np.hstack((
                        Phi_H_2D_y(self.X_N, self.Y_N, self.n_hb),
                        Phi_RBF_2D_y(self.X_N, self.Y_N,
                                      self.X_C, self.Y_C,
                                      self.c_k, self.basis)
                        ))
                    # Compute Phi_n on X_N (equation (18))
                    Matrix_Phi_N = Matrix_Phi_2D_X_N_der_x*self.n_x[:, np.newaxis] +\
                                    Matrix_Phi_2D_X_N_der_y*self.n_y[:, np.newaxis]
                else: # No Neumann conditions
                    # initialize the empty array
                    Matrix_Phi_N = np.empty((0,self.n_b))
                # block structure as in equation (17)
                Matrix_D_N = np.block([
                    [Matrix_Phi_N,np.zeros((self.n_N, self.n_b))],
                    [np.zeros((self.n_N, self.n_b)), Matrix_Phi_N]
                    ])
                
                # We can now assemble the matrix independent of what combinations
                # of Dirichlet and Neumann we have
                self.B = np.vstack((Matrix_D_nabla, Matrix_D, Matrix_D_N)).T
                # We do the same for b_2, as this can also be done for every case
                self.b_2 = np.concatenate((np.zeros(self.n_Div),self.c_D_X, self.c_D_Y,
                                          self.c_N_X, self.c_N_Y)) / self.rescale
                
                # We compute Phi on all node points X
                Matrix_Phi_2D_X = np.hstack((
                    Phi_H_2D(self.X_G, self.Y_G, self.n_hb),
                    Phi_RBF_2D(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # block structure of A as in equation(10)
                PhiT_dot_Phi = Matrix_Phi_2D_X.T.dot(Matrix_Phi_2D_X)
                self.A = 2*np.block([
                    [PhiT_dot_Phi, np.zeros((self.n_b, self.n_b))],
                    [np.zeros((self.n_b, self.n_b)), PhiT_dot_Phi]
                    ])
                # compute b_1
                self.b_1 = 2*np.concatenate((Matrix_Phi_2D_X.T.dot(self.u), Matrix_Phi_2D_X.T.dot(self.v))) / self.rescale
                
            
                # if alpha_div is not None, we have a divergence free penalty
                # in the entire domain which we must add to the A matrix
                if alpha_div is not None:    
                    # Compute Phi_x on X_Div
                    Matrix_Phi_2D_X_der_x = np.hstack((
                        Phi_H_2D_x(self.X_G, self.Y_G, self.n_hb),
                        Phi_RBF_2D_x(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_Div
                    Matrix_Phi_2D_X_der_y = np.hstack((
                        Phi_H_2D_y(self.X_G, self.Y_G, self.n_hb),
                        Phi_RBF_2D_y(self.X_G, self.Y_G, self.X_C, self.Y_C, self.c_k, self.basis)
                        ))  
                    
                    # Compute the individual matrix products between x, y and z
                    # For the diagonal
                    PhiXT_dot_PhiX = Matrix_Phi_2D_X_der_x.T.dot(Matrix_Phi_2D_X_der_x)
                    PhiYT_dot_PhiY = Matrix_Phi_2D_X_der_y.T.dot(Matrix_Phi_2D_X_der_y) 
                    # For the off-diagonal elements
                    PhiXT_dot_PhiY = Matrix_Phi_2D_X_der_x.T.dot(Matrix_Phi_2D_X_der_y)
                    
                    # Diagonal
                    self.A[self.n_b*0:self.n_b*1,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiX
                    self.A[self.n_b*1:self.n_b*2,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiYT_dot_PhiY
                    # Upper off-diagonal elements
                    self.A[self.n_b*0:self.n_b*1,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiXT_dot_PhiY
                    # Lower off-diagonal elements
                    self.A[self.n_b*1:self.n_b*2,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiY.T 
                   
                    
                   
            elif self.type == '3D': # 3D
                # define the rescaling factor which is done based on the maximum
                # absolute velocity that is available in u and v
                data = np.concatenate((self.u, self.v, self.w))
                self.rescale = data[np.argmax(np.abs(data))]
                
                # Here, we compute the matrix D_nabla as in equation (15)
                # There is no check whether X_Div as it is assumed that the laminar
                # case is only used to enforce divergence free constraints
                # TODO implement a check that gives an error when laminar is called
                # and no divergence free constraints are used
                
                # compute the derivatives in x
                Matrix_Phi_3D_X_nabla_der_x = np.hstack((
                    Phi_H_3D_x(self.X_Div, self.Y_Div, self.Z_Div, self.n_hb),
                    Phi_RBF_3D_x(self.X_Div, self.Y_Div, self.Z_Div,
                                 self.X_C, self.Y_C, self.Z_C,
                                 self.c_k, self.basis)
                    ))
                # compute the derivatives in y
                Matrix_Phi_3D_X_nabla_der_y = np.hstack((
                    Phi_H_3D_y(self.X_Div, self.Y_Div, self.Z_Div, self.n_hb),
                    Phi_RBF_3D_y(self.X_Div, self.Y_Div, self.Z_Div,
                                 self.X_C, self.Y_C, self.Z_C,
                                 self.c_k, self.basis)
                    ))
                # compute the derivatives in z
                Matrix_Phi_3D_X_nabla_der_z = np.hstack((
                    Phi_H_3D_z(self.X_Div, self.Y_Div, self.Z_Div, self.n_hb),
                    Phi_RBF_3D_z(self.X_Div, self.Y_Div, self.Z_Div,
                                 self.X_C, self.Y_C, self.Z_C,
                                 self.c_k, self.basis)
                    ))
                # stack them together to obtain X_nabla
                Matrix_D_nabla = np.hstack((Matrix_Phi_3D_X_nabla_der_x,
                                            Matrix_Phi_3D_X_nabla_der_y,
                                            Matrix_Phi_3D_X_nabla_der_z)) 
                
                # For the Neumann and Dirichlet B.C., we do the check whether 
                # we have them or not. If we have them, the matrices are computed
                # If not, we initialize an empty array of appropriate size. This
                # allows us to stack them together universally without checking all
                # four possibilities of having or not having the two
                
                # Check for Dirichlet
                if self.n_D !=0: # We have Dirichlet conditions
                    # Compute Phi on X_D (16)
                    Matrix_Phi_3D_D = np.hstack((
                        Phi_H_3D(self.X_D, self.Y_D, self.Z_D, self.n_hb),
                        Phi_RBF_3D(self.X_D, self.Y_D, self.Z_D,
                                   self.X_C, self.Y_C, self.Z_C,
                                   self.c_k, self.basis)
                        ))
                else: # No Dirichlet conditions
                    # initialize the empty array
                    Matrix_Phi_3D_D = np.empty((0, self.n_b))
                # stack into the block structure of equation (16)
                Matrix_D = np.block([
                    [Matrix_Phi_3D_D, np.zeros((self.n_D, self.n_b)), np.zeros((self.n_D, self.n_b))],
                    [np.zeros((self.n_D, self.n_b)), Matrix_Phi_3D_D, np.zeros((self.n_D, self.n_b))],
                    [np.zeros((self.n_D, self.n_b)), np.zeros((self.n_D, self.n_b)), Matrix_Phi_3D_D]
                    ])

                # Check for Neumann
                if self.n_N != 0: # We have Neumann conditions
                    # Compute Phi_x on X_N
                    Matrix_Phi_3D_X_N_der_x = np.hstack((
                        Phi_H_3D_x(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_x(self.X_N, self.Y_N, self.Z_N,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_N
                    Matrix_Phi_3D_X_N_der_y = np.hstack((
                        Phi_H_3D_y(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_y(self.X_N, self.Y_N, self.Z_N,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_z on X_N
                    Matrix_Phi_3D_X_N_der_z = np.hstack((
                        Phi_H_3D_z(self.X_N, self.Y_N, self.Z_N, self.n_hb),
                        Phi_RBF_3D_z(self.X_N, self.Y_N, self.Z_N,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_n on X_N (equation (18))
                    Matrix_Phi_N = Matrix_Phi_3D_X_N_der_x*self.n_x[:, np.newaxis] +\
                                   Matrix_Phi_3D_X_N_der_y*self.n_y[:, np.newaxis] +\
                                   Matrix_Phi_3D_X_N_der_z*self.n_z[:, np.newaxis]
                else: # No Neumann conditions
                    # initialize the empty array
                    Matrix_Phi_N = np.empty((0,self.n_b))
                # block structure as in equation (17)
                Matrix_D_N = np.block([
                    [Matrix_Phi_N, np.zeros((self.n_N, self.n_b)), np.zeros((self.n_N, self.n_b))],
                    [np.zeros((self.n_N, self.n_b)), Matrix_Phi_N, np.zeros((self.n_N, self.n_b))],
                    [np.zeros((self.n_N, self.n_b)), np.zeros((self.n_N, self.n_b)), Matrix_Phi_N]
                    ])

                # We can now assemble the matrix regardless of what combinations
                # of Dirichlet and Neumann we have
                self.B = np.vstack((Matrix_D_nabla, Matrix_D, Matrix_D_N)).T
                # We do the same for b_2, as this can also be done for every case
                self.b_2 = np.concatenate((np.zeros(self.n_Div),
                                          self.c_D_X, self.c_D_Y, self.c_D_Z,
                                          self.c_N_X, self.c_N_Y, self.c_N_Z)) / self.rescale

                # We compute Phi on all node points X
                Matrix_Phi_3D_X = np.hstack((
                    Phi_H_3D(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                    Phi_RBF_3D(self.X_G, self.Y_G, self.Z_G,
                               self.X_C, self.Y_C, self.Z_C,
                               self.c_k, self.basis)
                    ))
                # block structure of A as in equation(10)
                PhiT_dot_Phi = Matrix_Phi_3D_X.T.dot(Matrix_Phi_3D_X)
                self.A = 2*np.block([
                    [PhiT_dot_Phi, np.zeros((self.n_b, self.n_b)), np.zeros((self.n_b, self.n_b))],
                    [np.zeros((self.n_b, self.n_b)), PhiT_dot_Phi, np.zeros((self.n_b, self.n_b))],
                    [np.zeros((self.n_b, self.n_b)), np.zeros((self.n_b, self.n_b)), PhiT_dot_Phi]
                    ])
                # compute b_1
                self.b_1 = 2*np.concatenate((Matrix_Phi_3D_X.T.dot(self.u),
                                             Matrix_Phi_3D_X.T.dot(self.v),
                                             Matrix_Phi_3D_X.T.dot(self.w))) / self.rescale
                
                # if alpha_div is not None, we have a divergence free penalty
                # in the entire domain which we must add to the A matrix
                if alpha_div is not None:    
                    # Compute Phi_x on X_Div
                    Matrix_Phi_3D_X_der_x = np.hstack((
                        Phi_H_3D_x(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                        Phi_RBF_3D_x(self.X_G, self.Y_G, self.Z_G,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))
                    # Compute Phi_y on X_Div
                    Matrix_Phi_3D_X_der_y = np.hstack((
                        Phi_H_3D_y(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                        Phi_RBF_3D_y(self.X_G, self.Y_G, self.Z_G,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))  
                    # Compute Phi_z on X_Div
                    Matrix_Phi_3D_X_der_z = np.hstack((
                        Phi_H_3D_z(self.X_G, self.Y_G, self.Z_G, self.n_hb),
                        Phi_RBF_3D_z(self.X_G, self.Y_G, self.Z_G,
                                     self.X_C, self.Y_C, self.Z_C,
                                     self.c_k, self.basis)
                        ))  
                    
                    # Compute the individual matrix products between x, y and z
                    # For the diagonal
                    PhiXT_dot_PhiX = Matrix_Phi_3D_X_der_x.T.dot(Matrix_Phi_3D_X_der_x)
                    PhiYT_dot_PhiY = Matrix_Phi_3D_X_der_y.T.dot(Matrix_Phi_3D_X_der_y) 
                    PhiZT_dot_PhiZ = Matrix_Phi_3D_X_der_z.T.dot(Matrix_Phi_3D_X_der_z) 
                    # For the off-diagonal elements
                    PhiXT_dot_PhiY = Matrix_Phi_3D_X_der_x.T.dot(Matrix_Phi_3D_X_der_y)
                    PhiXT_dot_PhiZ = Matrix_Phi_3D_X_der_x.T.dot(Matrix_Phi_3D_X_der_z)
                    PhiYT_dot_PhiZ = Matrix_Phi_3D_X_der_y.T.dot(Matrix_Phi_3D_X_der_z)
                    
                    # Diagonal
                    self.A[self.n_b*0:self.n_b*1,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiX
                    self.A[self.n_b*1:self.n_b*2,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiYT_dot_PhiY
                    self.A[self.n_b*2:self.n_b*3,self.n_b*2:self.n_b*3] += 2*alpha_div*PhiZT_dot_PhiZ
                    
                    # Upper off-diagonal elements
                    self.A[self.n_b*0:self.n_b*1,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiXT_dot_PhiY
                    self.A[self.n_b*0:self.n_b*1,self.n_b*2:self.n_b*3] += 2*alpha_div*PhiXT_dot_PhiZ
                    self.A[self.n_b*1:self.n_b*2,self.n_b*2:self.n_b*3] += 2*alpha_div*PhiYT_dot_PhiZ
                    
                    # Lower off-diagonal elements
                    self.A[self.n_b*1:self.n_b*2,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiY.T
                    self.A[self.n_b*2:self.n_b*3,self.n_b*0:self.n_b*1] += 2*alpha_div*PhiXT_dot_PhiZ.T
                    self.A[self.n_b*2:self.n_b*3,self.n_b*1:self.n_b*2] += 2*alpha_div*PhiYT_dot_PhiZ.T
                
        elif self.model == 'RANSI':  
            raise NotImplementedError('RANSI currently not implemented')
        elif self.model == 'RANSI':  
            raise NotImplementedError('RANSI currently not implemented')
        else:
            raise ValueError('No regression could be performed, check that the model is correctly set')
        return
    
   
        

    # 5 Solver using the Shur complement
    def Solve(self, K_cond=1e12):
        """
        This function solves the constrained quadratic problem A, B, b_1, b_2.
        The method is universal for 2D/3D problems as well as laminar/poisson problems
    
        The input parameters are the class itself and the desired condition 
        number of A which is fixed based on its largest and smallest eigenvalue
        
        The function assigns the weights 'w' and the Lagrange multipliers
        Lambda to the class. The weights are computed for the min/max scaled problem,
        i.e. the right hand-side of the linear system is normalized. The assigned
        weights are rescaled by self.rescale to get the real, physical quantities
        
        TODO Suggestion: We do a check whether B is empty and if it is, just 
        do the solution based on the inverted A. This would allow a regression
        without any constraints
        
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param K_cond: float
          This is the regularization parameter. It is fixing the condition number
          The estimation is based such that the regularize matrix has the condition
          number k_cond. For this, we compute the max and the min eigenvalue.
        """   
    
        # Two options: 
        # 1.: We have constraints, then B and b_2 are not empty and we go for Schur complements
        # 2.: We do not have constraints, then we only need to solve A*w = b_1
        if (self.B.size == 0) and (self.b_2.size == 0):
            print('Solving without constriaints')
            
            # Step 1: Regularize the matrix A
            lambda_A = eigsh(self.A, 1, return_eigenvectors=False) # Largest eigenvalue
            alpha = lambda_A / K_cond
            self.A = self.A + alpha*np.eye(np.shape(self.A)[0])
            print('Matrix A regularized')
            
            # Step 2: Cholesky Decomposition of A 
            L_A, low = linalg.cho_factor(self.A, overwrite_a = True, check_finite = False, lower = True)
            
            # Step 3: Solve for w
            self.weights = linalg.cho_solve((L_A, low), self.b_1, check_finite = False) * self.rescale
            
            
        elif (self.B.size != 0) and (self.b_2.size != 0):
            print('Solving with constriaints')
            
            # Step 1: Regularize the matrix A
            lambda_M = eigsh(self.A, 1, return_eigenvectors=False) # Largest eigenvalue
            alpha = lambda_M / K_cond
            self.A = self.A + alpha*np.eye(np.shape(self.A)[0])
            print('Matrix A regularized')
            
            # Step 2: Cholesky Decomposition of A    
            L_A, low = linalg.cho_factor(self.A, overwrite_a=True, check_finite=False, lower=True)
            
            # Step 3: Solve for N
            N = linalg.cho_solve((L_A,low), self.B, check_finite=False)
            
            # Step 4: prepare M 
            M = N.T@self.B
            
            # Step 5 + 6: Regularize M if needed, then compute chol factor
            try: 
                # try without regularization
                L_M, low = linalg.cho_factor(M, overwrite_a=True, check_finite=False, lower=True)
                print('Chol factor of M WITHOUT regularization')             
            except:
                # if it does not work, regularize M the same way as for A
                lambda_M = eigsh(M, 1, return_eigenvectors=False) # Largest eigenvalue
                alpha = lambda_M / K_cond
                M = M + alpha*np.eye(np.shape(M)[0])
                L_M, low = linalg.cho_factor(M, overwrite_a = True, check_finite = False, lower = True)
                print('Chol factor of M WITH regularization')             
           
            # Step 7: Solve the system for lambda    
            b2star = N.T.dot(self.b_1) - self.b_2
            self.lambdas = linalg.cho_solve((L_M, low), b2star, check_finite = False)
            print('Lambdas computed')
        
            # Step 8: Solve for w.
            b1_star = self.b_1 - self.B.dot(self.lambdas)
            self.weights= linalg.cho_solve((L_A, low), b1_star, check_finite=False) * self.rescale
            print('w computed')
        else:
            raise ValueError('b_1 or B is empty while the other is not, check your constraints!')
        return 



    # 6. Evaluate solution on arbitrary grid
    
    # Here is a function to compute the solution on an arbitrary set of points
    # X_G, Y_G. We take w, lam from the solution, X_C, Y_C, c_k from the clustering.
    def Get_Sol(self,grid):
        """
        This function evaluates the solution of the linear system on an arbitrary
        set of points on the grid.
        The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param grid: list 
            Contains the points at which the source term is evaluated
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].
            
        :return: U_P, V_P, W_P.  
        Depending on model='scalar/laminar' and type='2D/3D'    
        If scalar, the solution is only U_P.
        If laminar and 2D, the solution is U_P, V_P
        If laminar and 3D, the solution is U_P, V_P and W_P
        
        """   
        
        # Check the input is correct
        assert type(grid) == list, 'grid must be a list'
        
                
        # check if this instance has a 2D 
        if len(grid) == 2 and self.type == '2D': # 2D 
            # Assign the grid
            X_P = grid[0]
            Y_P = grid[1]
            # number of points on the new grid
            n_p = X_P.shape[0]
            
            # Check what model type we have
            if self.model == 'scalar': # Scalar
                # Evaluate Phi on the grid X_P
                Phi=np.hstack((
                    Phi_H_2D(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))  
                # Compute U on the new grid
                U_P=Phi.dot(self.weights)
                return U_P
                
            elif self.model == 'laminar': # Laminar
                # Evaluate Phi on the grid X_P
                Phi_Sub=np.hstack((
                    Phi_H_2D(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                U_P = U_f[:n_p]
                V_P = U_f[n_p:]                
                
                return U_P, V_P
                
        elif len(grid) == 3 and self.type == '3D': # 3D
            # Assign the grid
            X_P = grid[0]
            Y_P = grid[1]
            Z_P = grid[2]
            # number of points on the new grid
            n_p = X_P.shape[0]
            
            # Check what model type we have
            if self.model == 'scalar': # Scalar
                # Evaluate Phi on the grid X_P
                Phi=np.hstack((
                    Phi_H_3D(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))  
                # Compute U on the new grid
                U_P=Phi.dot(self.weights)
                return U_P
                
            elif self.model == 'laminar': # Laminar
                # Evaluate Phi on the grid X_P
                Phi_Sub=np.hstack((
                    Phi_H_3D(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                U_P = U_f[0*n_p:1*n_p]
                V_P = U_f[1*n_p:2*n_p]              
                W_P = U_f[2*n_p:3*n_p]
                return U_P, V_P, W_P
                
                
        else:
            raise ValueError('Length of Grid is invalid for Type ' + self.type)
            
       



    def Get_first_Derivatives(self,grid):
        ''' Compute all derivatives from the RBF solution 
        The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param grid: list 
            Contains the points at which the source term is evaluated
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].

        :return: dudx, dudy, dudx/ dvdx, dvdy, dvdx/ dwdx, dwdy, dwdx.
        
        Depending on model='scalar/laminar' and type='2D/3D'    
        If scalar and 2D, the output is dudx, dudy
        If scalar and 3D, the output is dudx, dudy, dudz        
        If laminar and 2D, the solution is dudx, dudy, dvdx, dvdy
        If laminar and 3D, the solution is dudx,dudy,dudx,dvdx,dvdy,dvdx,dwdx,dwdy,dwdx, 
            
        ''' 
        
        # Check the input is correct
        assert type(grid) == list, 'grid must be a list'
        
                
        # check if this instance has a 2D 
        if len(grid) == 2 and self.type == '2D': # 2D 
            # Assign the grid
            X_P = grid[0]
            Y_P = grid[1]
            # number of points on the new grid
            n_p = X_P.shape[0]
            
            # Check what model type we have
            if self.model == 'scalar': # Scalar
                # Evaluate Phi_x on the grid X_P,Y_P
                Phi_x=np.hstack((
                    Phi_H_2D_x(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_x(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))  
                # Evaluate Phi_y on the grid X_P,Y_P
                Phi_y=np.hstack((
                    Phi_H_2D_y(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_y(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))                 
                # Compute dudx and dudy on the new grid
                dudx=Phi_x.dot(self.weights)
                dudy=Phi_y.dot(self.weights)
                return dudx, dudy
                
            elif self.model == 'laminar': # Laminar
                # We do it in 2 blocks: first all derivatives in x
                # Evaluate Phi on the grid X_P, Y_P
                Phi_Sub=np.hstack((
                    Phi_H_2D_x(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_x(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                dudx = U_f[:n_p]
                dvdx = U_f[n_p:]     
                
                # Then we do it again for the derivatives of y.
                # Note however, that we re-use the same variables Phi_Sub and Phi
                # to limit the memory usage. This is pretty much copy-paste.
                Phi_Sub=np.hstack((
                    Phi_H_2D_y(X_P, Y_P, self.n_hb),
                    Phi_RBF_2D_y(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                    ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                dudy = U_f[:n_p]
                dvdy = U_f[n_p:]     

                return dudx,dudy,dvdx,dvdy
                
        elif len(grid) == 3 and self.type == '3D': # 3D
            # Assign the grid
            X_P = grid[0]
            Y_P = grid[1]
            Z_P = grid[2]
            # number of points on the new grid
            n_p = X_P.shape[0]
            
            # Check what model type we have
            if self.model == 'scalar': # Scalar
                # Evaluate Phi derivatives on the grid X_P,Y_P, Z_P.
                # We repeat the calculation of the three derivatives using
                # the same matrices. Here is for dudx
                Phi=np.hstack((
                    Phi_H_3D_x(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D_x(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))  
                # Compute dudx on the new grid
                dudx=Phi.dot(self.weights)
                # Now again for the derivative on dudy
                Phi=np.hstack((
                    Phi_H_3D_y(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D_y(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))  
                # Compute dudx on the new grid
                dudy=Phi.dot(self.weights)                
                # Now again for the derivative on dudz
                Phi=np.hstack((
                    Phi_H_3D_z(X_P, Y_P, Z_P, self.n_hb),
                    Phi_RBF_3D_z(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                    ))  
                # Compute dudx on the new grid
                dudz=Phi.dot(self.weights)                  
                return dudx,dudy,dudz
                
            elif self.model == 'laminar': # Laminar
                # Evaluate Phi derivatives on the grid X_P,Y_P, Z_P. 
                # We repeat it 3 times for each derivative and stack
                # all components into one single vector at each step
                # This can be improved later. We use always the same
                # matrices for memory efficiency
                    
                # All derivatives along x -------------------------
                Phi_Sub=np.hstack((
                     Phi_H_3D_x(X_P, Y_P, Z_P, self.n_hb),
                     Phi_RBF_3D_x(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                     ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                dudx = U_f[0*n_p:1*n_p]
                dvdx = U_f[1*n_p:2*n_p]              
                dwdx = U_f[2*n_p:3*n_p]
                
                # All derivatives along y -------------------------
                Phi_Sub=np.hstack((
                     Phi_H_3D_y(X_P, Y_P, Z_P, self.n_hb),
                     Phi_RBF_3D_y(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                     ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                dudy = U_f[0*n_p:1*n_p]
                dvdy = U_f[1*n_p:2*n_p]              
                dwdy = U_f[2*n_p:3*n_p]
                
                # All derivatives along z -------------------------
                Phi_Sub=np.hstack((
                     Phi_H_3D_z(X_P, Y_P, Z_P, self.n_hb),
                     Phi_RBF_3D_z(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                     ))
                # Create the block structure of equation (16)
                Phi = np.block([
                    [Phi_Sub, np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), Phi_Sub, np.zeros((n_p, self.n_b))],
                    [np.zeros((n_p, self.n_b)), np.zeros((n_p, self.n_b)), Phi_Sub]
                    ])
                # compute the solution
                U_f=Phi.dot(self.weights)
                dudz = U_f[0*n_p:1*n_p]
                dvdz = U_f[1*n_p:2*n_p]              
                dwdz = U_f[2*n_p:3*n_p]
                
                
                return dudx,dudy,dudz,dvdx,dvdy,dvdz,dwdx,dwdy,dwdz
                
                
        else:
            raise ValueError('Length of Grid is invalid for Type ' + self.type)
         
        
   

    # Here is a function to evaluate the forcing term on the grid points that are 
    # used for the pressure
    def Evaluate_Source_Term(self, grid, rho):
        """
        This function evaluates the source term on the right hand side of
        equation (21)
    
        The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param grid: list
            Contains the points at which the source term is evaluated
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].
        
       :param rho: float
           Density of the fluid.
           
        Returns
        -------
        :param source_term: 1d np.array
            R.h.s. of equation (21)
        """
        
        # Check the input is correct
        assert type(grid) == list, 'grid must be a list'
        
        # check whether it is 2D or 3D
        if len(grid) == 2 and self.type == '2D': # 2D
            # assign the grid points in X and Y
            X_P = grid[0]
            Y_P = grid[1]
            W_u = self.weights[:self.n_b]
            W_v = self.weights[self.n_b:]
            
            # We compute Phi_x on X_P
            Matrix_Phi_2D_X_P_der_x = np.hstack((
                Phi_H_2D_x(X_P, Y_P, self.n_hb),
                Phi_RBF_2D_x(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along x
            dUdX = Matrix_Phi_2D_X_P_der_x.dot(W_u)
            dVdX = Matrix_Phi_2D_X_P_der_x.dot(W_v)
            
            # We compute Phi_y on X_P
            Matrix_Phi_2D_X_P_der_y = np.hstack((
                Phi_H_2D_y(X_P, Y_P, self.n_hb),
                Phi_RBF_2D_y(X_P, Y_P, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along y
            dUdY = Matrix_Phi_2D_X_P_der_y.dot(W_u)
            dVdY = Matrix_Phi_2D_X_P_der_y.dot(W_v)
        
            #forcing term is evaluated
            source_term = -rho*(dUdX**2+2*dUdY*dVdX+dVdY**2)
            
        elif len(grid) == 3 and self.type == '3D':
            # assign the grid points in X and Y
            X_P = grid[0]
            Y_P = grid[1]
            Z_P = grid[2]
            W_u = self.weights[0*self.n_b:1*self.n_b]
            W_v = self.weights[1*self.n_b:2*self.n_b]
            W_w = self.weights[2*self.n_b:3*self.n_b]
            
            # We compute Phi_x on X_P
            Matrix_Phi_3D_X_P_der_x = np.hstack((
                Phi_H_3D_x(X_P, Y_P, Z_P, self.n_hb),
                Phi_RBF_3D_x(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along x
            dUdX = Matrix_Phi_3D_X_P_der_x.dot(W_u)
            dVdX = Matrix_Phi_3D_X_P_der_x.dot(W_v)
            dWdX = Matrix_Phi_3D_X_P_der_x.dot(W_w)
            
            # We compute Phi_y on X_P
            Matrix_Phi_3D_X_P_der_y = np.hstack((
                Phi_H_3D_y(X_P, Y_P, Z_P, self.n_hb),
                Phi_RBF_3D_y(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along y
            dUdY = Matrix_Phi_3D_X_P_der_y.dot(W_u)
            dVdY = Matrix_Phi_3D_X_P_der_y.dot(W_v)
            dWdY = Matrix_Phi_3D_X_P_der_y.dot(W_w)
            
            # We compute Phi_z on X_P
            Matrix_Phi_3D_X_P_der_z = np.hstack((
                Phi_H_3D_z(X_P, Y_P, Z_P, self.n_hb),
                Phi_RBF_3D_z(X_P, Y_P, Z_P, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # We compute the derivatives of the velocity field along y
            dUdZ = Matrix_Phi_3D_X_P_der_z.dot(W_u)
            dVdZ = Matrix_Phi_3D_X_P_der_z.dot(W_v)
            dWdZ = Matrix_Phi_3D_X_P_der_z.dot(W_w)
        
            #forcing term is evaluated
            source_term = -rho*(dUdX**2+dVdY**2+dWdZ**2 + 2*dUdY*dVdX + 2*dUdZ*dWdX + 2*dVdZ*dWdY)
            
        else:
            raise ValueError('Length of Grid is invalid for Type ' + self.type)
        
        return source_term
    
    def Get_Pressure_Neumann(self, grid, normals, rho, mu):
        """
        This function evaluates the Neumann boundary conditions for the pressure
        integration in equation (29).
        
        The input parameters are 
        ------------------------------------------------------------------------
        Parameters
        ------------------------------------------------------------------------
        :param grid: list
            Contains the points at which the source term is evaluated.
            If the model is 2D, then this has [X_P, Y_P].
            If the model is 3D, then this has [X_P, Y_P, Z_P].
        
        :param normals: list
            Contains the points at which the source term is evaluated.
            If the model is 2D, then this has [n_x, n_y].
            If the model is 3D, then this has [n_x, n_y, n_z].
            
        :param rho: float
            Density of the fluid.
        :param mu: float
            Dynamic viscosity of the fluid.
            
        Returns
        -------
        :param P_neu: 1d np.array
            Normal pressure in equation (29).
        """
        
        # Check the input is correct
        assert type(grid) == list, 'grid must be a list'
        assert type(normals) == list, 'normals must be a list'
        assert len(grid) == len(normals), 'Length of grid must be equal to the length of normals'
        # Check if we have 2D or 3D data
        if len(grid) == 2 and self.type == '2D': # 2D
            # Assign the grid
            X_N = grid[0]
            Y_N = grid[1]
            # Assign the normals
            n_x = normals[0]
            n_y = normals[1]
            # Assign the weights
            W_u = self.weights[:self.n_b]
            W_v = self.weights[self.n_b:]
            # Compute the matrix Phi_x on X_N
            Matrix_Phi_2D_X_N_der_x = np.hstack((
                Phi_H_2D_x(X_N, Y_N, self.n_hb),
                Phi_RBF_2D_x(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along x
            dUdX = Matrix_Phi_2D_X_N_der_x.dot(W_u)
            dVdX = Matrix_Phi_2D_X_N_der_x.dot(W_v)
            
            # Compute the matrix Phi_y on X_N
            Matrix_Phi_2D_X_N_der_y = np.hstack((
                Phi_H_2D_y(X_N, Y_N, self.n_hb),
                Phi_RBF_2D_y(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along y
            dUdY = Matrix_Phi_2D_X_N_der_y.dot(W_u)
            dVdY = Matrix_Phi_2D_X_N_der_y.dot(W_v)
            
            # Compute the matrix Phi on X_N
            Matrix_Phi_2D_X_N = np.hstack((
                Phi_H_2D(X_N, Y_N, self.n_hb),
                Phi_RBF_2D(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the velocities
            U = Matrix_Phi_2D_X_N.dot(W_u)
            V = Matrix_Phi_2D_X_N.dot(W_v)
            
            # Compute the Laplacian on X_N
            L_X_N = np.hstack((
                Phi_H_2D_Laplacian(X_N, Y_N, self.n_hb),
                Phi_RBF_2D_Laplacian(X_N, Y_N, self.X_C, self.Y_C, self.c_k, self.basis)
                ))
            # Compute the Laplacian for U and V
            L_U = L_X_N.dot(W_u)
            L_V = L_X_N.dot(W_v)
            
            # Compute the pressure normals
            P_N_x = mu*L_U - rho * (U*dUdX + V*dUdY)
            P_N_y = mu*L_V - rho * (U*dVdX + V*dVdY)
            
            # Multiply with the normals to get the projected pressure
            P_Neu = P_N_x * n_x + P_N_y * n_y
            
        elif len(grid) == 3 and self.type == '3D':
            # Assign the grid
            X_N = grid[0]
            Y_N = grid[1]
            Z_N = grid[2]
            # Assign the normals
            n_x = normals[0]
            n_y = normals[1]
            n_z = normals[2]
            # Assign the weights
            W_u = self.weights[0*self.n_b:1*self.n_b]
            W_v = self.weights[1*self.n_b:2*self.n_b]
            W_w = self.weights[2*self.n_b:3*self.n_b]
            # Compute the matrix Phi_x on X_N
            Matrix_Phi_3D_X_N_der_x = np.hstack((
                Phi_H_3D_x(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_x(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along x
            dUdX = Matrix_Phi_3D_X_N_der_x.dot(W_u)
            dVdX = Matrix_Phi_3D_X_N_der_x.dot(W_v)
            dWdX = Matrix_Phi_3D_X_N_der_x.dot(W_w)
            
            # Compute the matrix Phi_y on X_N
            Matrix_Phi_3D_X_N_der_y = np.hstack((
                Phi_H_3D_y(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_y(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along y
            dUdY = Matrix_Phi_3D_X_N_der_y.dot(W_u)
            dVdY = Matrix_Phi_3D_X_N_der_y.dot(W_v)
            dWdY = Matrix_Phi_3D_X_N_der_y.dot(W_w)
            
            # Compute the matrix Phi_z on X_N
            Matrix_Phi_3D_X_N_der_z = np.hstack((
                Phi_H_3D_z(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_z(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the derivatives along y
            dUdZ = Matrix_Phi_3D_X_N_der_z.dot(W_u)
            dVdZ = Matrix_Phi_3D_X_N_der_z.dot(W_v)
            dWdZ = Matrix_Phi_3D_X_N_der_z.dot(W_w)
            
            # Compute the matrix Phi on X_N
            Matrix_Phi_3D_X_N = np.hstack((
                Phi_H_3D(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the velocities
            U = Matrix_Phi_3D_X_N.dot(W_u)
            V = Matrix_Phi_3D_X_N.dot(W_v)
            W = Matrix_Phi_3D_X_N.dot(W_w)
            
            # Compute the Laplacian on X_N
            L_X_N = np.hstack((
                Phi_H_3D_Laplacian(X_N, Y_N, Z_N, self.n_hb),
                Phi_RBF_3D_Laplacian(X_N, Y_N, Z_N, self.X_C, self.Y_C, self.Z_C, self.c_k, self.basis)
                ))
            # Compute the Laplacian for U and V
            L_U = L_X_N.dot(W_u)
            L_V = L_X_N.dot(W_v)
            L_W = L_X_N.dot(W_w)
            
            # Compute the pressure normals
            P_N_x = mu*L_U - rho * (U*dUdX + V*dUdY + W*dUdZ)
            P_N_y = mu*L_V - rho * (U*dVdX + V*dVdY + W*dVdZ)
            P_N_z = mu*L_W - rho * (U*dWdX + V*dWdY + W*dWdZ)
            
            # Multiply with the normals to get the projected pressure
            P_Neu = P_N_x * n_x + P_N_y * n_y + P_N_z * n_z
            
        else:
            raise ValueError('Length of Grid is invalid for Type ' + self.type)
            
        return P_Neu


# =============================================================================
#  Utilities functions
# =============================================================================

# =============================================================================
#  RBF functions in 2D
#  Includes: Phi_RBF_2D, Phi_RBF_2D_x, Phi_RBF_2D_y, Phi_RBF_2D_Laplacian
# =============================================================================

def Phi_RBF_2D(X_G, Y_G, X_C, Y_C, c_k, basis):
    """
    Get the basis matrix at the points (X_G,Y_G) from RBFs at the collocation points
    at (X_C,Y_C), having shape factors c_k. The output is a matrix of side (n_p) x (n_c).
    The basis can be 'c4' or 'gauss'.
    """
    # This is the contribution of the RBF part
    n_b = len(X_C); n_p = len(X_G)
    Phi_RBF = np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2+(Y_C[r]-Y_G)**2))
            # Assemble into matrix
            Phi_RBF[:,r]=gaussian

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_G - X_C[r])**2 + (Y_G - Y_C[r])**2)
            # Compute Phi
            phi = (1 + d/c_k[r])**5 * (1 - d/c_k[r])**5
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF[:,r] = phi
    
    # Return the matrix
    return Phi_RBF


def Phi_RBF_2D_x(X_G, Y_G, X_C, Y_C, c_k, basis):
    """
    Get the derivative along x of the basis matrix at the points (X_G,Y_G) from 
    RBFs at the collocation points at (X_C,Y_C), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """
    # number of bases (n_b) and points (n_p)
    n_b = len(X_C); n_p = len(X_G)
    # Initialize the matrix
    Phi_RBF_x = np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2+(Y_C[r]-Y_G)**2))
            # Multiply with inner term and assemble into matrix
            Phi_RBF_x[:,r]=2*c_k[r]**2*(X_C[r]-X_G)*gaussian
            
    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_C[r] - X_G)**2 + (Y_C[r] - Y_G)**2)
            # Compute derivative along x
            phi = 10 / c_k[r]**10 * (c_k[r] + d)**4 * (c_k[r] - d)**4 * (X_C[r] - X_G)
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF_x[:,r] = phi

    # Return the matrix
    return Phi_RBF_x


def Phi_RBF_2D_y(X_G, Y_G, X_C, Y_C, c_k, basis):
    """
    Get the derivative along y of the basis matrix at the points (X_G,Y_G) from 
    RBFs at the collocation points at (X_C,Y_C), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """
    # number of bases (n_b) and points (n_p)
    n_b = len(X_C); n_p = len(X_G)
    # Initialize the matrix
    Phi_RBF_y = np.zeros((n_p,n_b))
  
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2+(Y_C[r]-Y_G)**2))
            # Multiply with inner term and assemble into matrix
            Phi_RBF_y[:,r]=2*c_k[r]**2*(Y_C[r]-Y_G)*gaussian
            
    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_G - X_C[r])**2 + (Y_G - Y_C[r])**2)
            # Compute derivative along y
            phi = 10 / c_k[r]**10 * (c_k[r] + d)**4 * (c_k[r] - d)**4 * (Y_C[r] - Y_G)
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF_y[:,r] = phi
            
    # Return the matrix
    return Phi_RBF_y


def Phi_RBF_2D_Laplacian(X_G, Y_G, X_C, Y_C, c_k, basis):
    """
    Get the Laplacian of the basis matrix at the points (X_G,Y_G) from 
    RBFs at the collocation points at (X_C,Y_C), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """ 
    # number of bases (n_b) and points (n_p)
    n_b = len(X_C); n_p = len(X_G)
    # Initialize the matrix    
    Lap_RBF = np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian = np.exp(-c_k[r]**2*((X_C[r]-X_G)**2+(Y_C[r]-Y_G)**2))
            # Get second derivative along x and y
            Partial_xx = 4*c_k[r]**4*(X_C[r]-X_G)**2*gaussian-2*c_k[r]**2*gaussian
            Partial_yy = 4*c_k[r]**4*(Y_C[r]-Y_G)**2*gaussian-2*c_k[r]**2*gaussian
            # Assemble into matrix
            Lap_RBF[:,r] = Partial_xx+Partial_yy 

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_G - X_C[r])**2 + (Y_G - Y_C[r])**2)
            # Compute the prefactor in the second derivative
            factor = 10 / c_k[r]**10 * (c_k[r] + d)**3 * (c_k[r] - d)**3
            # Multiply with inner derivative
            Partial_xx = factor * (8*(X_G - X_C[r])**2 - c_k[r]**2 + d**2)
            Partial_yy = factor * (8*(Y_G - Y_C[r])**2 - c_k[r]**2 + d**2)
            # Compute Laplacian
            Laplacian = Partial_xx + Partial_yy
            # Compact support
            Laplacian[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Lap_RBF[:,r] = Laplacian

    # Return the matrix
    return Lap_RBF


# =============================================================================
#  RBF functions in 3D
#  Includes: Phi_RBF_2D, Phi_RBF_2D_x, Phi_RBF_2D_y, Phi_RBF_2D_Laplacian
# =============================================================================
    
def Phi_RBF_3D(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
    """
    Get the basis matrix at the points (X_G,Y_G,Z_G) from RBFs at the collocation points
    at (X_C,Y_C,Z_C), having shape factors c_k. The output is a matrix of side (n_p) x (n_c).
    The basis can be 'c4' or 'gauss'.
    """
    # This is the contribution of the RBF part
    n_b=len(X_C); n_p=len(X_G)
    Phi_RBF=np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2 + (Y_C[r]-Y_G)**2 + (Z_C[r]-Z_G)**2))
            # Assemble into matrix
            Phi_RBF[:,r]=gaussian

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_G - X_C[r])**2 + (Y_G - Y_C[r])**2 + (Z_G - Z_C[r])**2)
            # Compute Phi
            phi = (1 + d/c_k[r])**5 * (1 - d/c_k[r])**5
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF[:,r] = phi
    
    return Phi_RBF


def Phi_RBF_3D_x(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
    """
    Get the derivative along x of the basis matrix at the points (X_G,Y_G,Z_G) from 
    RBFs at the collocation points at (X_C,Y_C,Z_G), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """
    # number of bases (n_b) and points (n_p)
    n_b=len(X_C); n_p=len(X_G)
    # Initialize the matrix
    Phi_RBF_x=np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2 + (Y_C[r]-Y_G)**2 + (Z_C[r]-Z_G)**2))
            # Multiply with inner term and assemble into matrix
            Phi_RBF_x[:,r]=2*c_k[r]**2*(X_C[r]-X_G)*gaussian
            
    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_C[r] - X_G)**2 + (Y_C[r] - Y_G)**2 + (Z_C[r] - Z_G)**2)
            # Compute derivative along x
            phi = 10 / c_k[r]**10 * (c_k[r] + d)**4 * (c_k[r] - d)**4 * (X_C[r] - X_G)
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF_x[:,r] = phi

    # Return the matrix
    return Phi_RBF_x


def Phi_RBF_3D_y(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
    """
    Get the derivative along y of the basis matrix at the points (X_G,Y_G,Z_G) from 
    RBFs at the collocation points at (X_C,Y_C,Z_G), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """
    # number of bases (n_b) and points (n_p)
    n_b = len(X_C); n_p = len(X_G)
    # Initialize the matrix
    Phi_RBF_y = np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2 + (Y_C[r]-Y_G)**2 + (Z_C[r]-Z_G)**2))
            # Multiply with inner term and assemble into matrix
            Phi_RBF_y[:,r]=2*c_k[r]**2*(Y_C[r]-Y_G)*gaussian
            
    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_C[r] - X_G)**2 + (Y_C[r] - Y_G)**2 + (Z_C[r] - Z_G)**2)
            # Compute derivative along x
            phi = 10 / c_k[r]**10 * (c_k[r] + d)**4 * (c_k[r] - d)**4 * (Y_C[r] - Y_G)
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF_y[:,r] = phi

    # Return the matrix
    return Phi_RBF_y


def Phi_RBF_3D_z(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
    """
    Get the derivative along z of the basis matrix at the points (X_G,Y_G,Z_G) from 
    RBFs at the collocation points at (X_C,Y_C,Z_G), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """
    # number of bases (n_b) and points (n_p)
    n_b = len(X_C); n_p = len(X_G)
    # Initialize the matrix
    Phi_RBF_z = np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian=np.exp(-c_k[r]**2*((X_C[r]-X_G)**2 + (Y_C[r]-Y_G)**2 + (Z_C[r]-Z_G)**2))
            # Multiply with inner term and assemble into matrix
            Phi_RBF_z[:,r]=2*c_k[r]**2*(Z_C[r]-Z_G)*gaussian
            
    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_C[r] - X_G)**2 + (Y_C[r] - Y_G)**2 + (Z_C[r] - Z_G)**2)
            # Compute derivative along x
            phi = 10 / c_k[r]**10 * (c_k[r] + d)**4 * (c_k[r] - d)**4 * (Z_C[r] - Z_G)
            # Compact support
            phi[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Phi_RBF_z[:,r] = phi

    # Return the matrix
    return Phi_RBF_z

def Phi_RBF_3D_Laplacian(X_G, Y_G, Z_G, X_C, Y_C, Z_C, c_k, basis):
    """
    Get the Laplacian of the basis matrix at the points (X_G,Y_G,Z_G) from 
    RBFs at the collocation points at (X_C,Y_C,Z_C), having shape factors c_k. The
    output is a matrix of side (n_p) x (n_c). The basis can be 'c4' or 'gauss'.
    """ 
    # number of bases (n_b) and points (n_p)
    n_b = len(X_C); n_p = len(X_G)
    # Initialize the matrix    
    Lap_RBF = np.zeros((n_p,n_b))
    
    # What comes next depends on the type of chosen RBF
    if basis == 'gauss':
        # Iterate over all basis elements
        for r in range(n_b):
            # Compute the Gaussian
            gaussian = np.exp(-c_k[r]**2*((X_C[r]-X_G)**2 + (Y_C[r]-Y_G)**2 + (Z_C[r]-Z_G)**2))
            # Get second derivative along x and y
            Partial_xx = 4*c_k[r]**4*(X_C[r]-X_G)**2*gaussian-2*c_k[r]**2*gaussian
            Partial_yy = 4*c_k[r]**4*(Y_C[r]-Y_G)**2*gaussian-2*c_k[r]**2*gaussian
            Partial_zz = 4*c_k[r]**4*(Z_C[r]-Z_G)**2*gaussian-2*c_k[r]**2*gaussian
            # Assemble into matrix
            Lap_RBF[:,r] = Partial_xx + Partial_yy  + Partial_zz

    elif basis == 'c4':
        # Iterate over all basis elements
        for r in range(n_b):
            # Get distance
            d = np.sqrt((X_G - X_C[r])**2 + (Y_G - Y_C[r])**2 + (Z_G - Z_C[r])**2)
            # Compute the prefactor in the second derivative
            factor = 10 / c_k[r]**10 * (c_k[r] + d)**3 * (c_k[r] - d)**3
            # Multiply with inner derivative
            Partial_xx = factor * (8*(X_G - X_C[r])**2 - c_k[r]**2 + d**2)
            Partial_yy = factor * (8*(Y_G - Y_C[r])**2 - c_k[r]**2 + d**2)
            Partial_zz = factor * (8*(Z_G - Z_C[r])**2 - c_k[r]**2 + d**2)
            # Compute Laplacian
            Laplacian = Partial_xx + Partial_yy + Partial_zz
            # Compact support
            Laplacian[np.abs(d) > c_k[r]] = 0
            # Assemble into matrix
            Lap_RBF[:,r] = Laplacian

    # Return the matrix
    return Lap_RBF

# =============================================================================
#  Harmonic functions in 2D
#  Includes: Phi_H_2D, Phi_H_2D_x, Phi_H_2D_y, Phi_H_2D_Laplacian
# =============================================================================

def Phi_H_2D(X_G, Y_G, n_hb):
    # Get the basis matrix at the points (X_G,Y_G) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p=len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Phi_H=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y

    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]            
        k_x_i=2*np.pi*(i+1) # This goes with sines
        k_x_j=np.pi/2*(2*j+1) # This goes with cosines
        k_y_m=2*np.pi*(m+1) # This goes with sines
        k_y_q=np.pi/2*(2*q+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x=np.sin(k_x_i*X_G); 
        cos_k_j_x=np.cos(k_x_j*X_G); 
        sin_k_m_y=np.sin(k_y_m*Y_G); 
        cos_k_q_y=np.cos(k_y_q*Y_G); 
                                
        # Assign the column of Phi_H
        Phi_H[:,count]=sin_k_i_x*cos_k_j_x*sin_k_m_y*cos_k_q_y
        count+=1  
    # # # Here's how to see these        
    # plt.scatter(X_G,Y_G,c=Phi_H[:,1])   
    return Phi_H


def Phi_H_2D_x(X_G,Y_G,n_hb):
    """
    Create the derivatives along x, Phi_x, for the n_hb harmonic bases, 
    computed on the points (X_G,Y_G)
    """
    # Get the number of points
    n_p=len(X_G)
    
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Phi_H_x=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y
    
    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]            
        k_x_i=2*np.pi*(i+1) # This goes with sines
        k_x_j=np.pi/2*(2*j+1) # This goes with cosines
        k_y_m=2*np.pi*(m+1) # This goes with sines
        k_y_q=np.pi/2*(2*q+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x=np.sin(k_x_i*X_G); cos_k_i_x=np.cos(k_x_i*X_G)
        cos_k_j_x=np.cos(k_x_j*X_G); sin_k_j_x=np.sin(k_x_j*X_G)
        sin_k_m_y=np.sin(k_y_m*Y_G); 
        cos_k_q_y=np.cos(k_y_q*Y_G); 
        # Assign the column of Phi_H
        Prime = -(k_x_j*sin_k_i_x*sin_k_j_x-k_x_i*cos_k_i_x*cos_k_j_x)   
        Phi_H_x[:,count] = Prime*sin_k_m_y*cos_k_q_y
        count+=1  
    return Phi_H_x


def Phi_H_2D_y(X_G,Y_G,n_hb):
    """
    Create the derivatives along y, Phi_y, for the n_hb harmonic bases, 
    computed on the points (X_G,Y_G)
    """
    # Get the number of points
    n_p=len(X_G)
    
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h=n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Phi_H_y=np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y
    
    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]           
        k_x_i=2*np.pi*(i+1) # This goes with sines
        k_x_j=np.pi/2*(2*j+1) # This goes with cosines
        k_y_m=2*np.pi*(m+1) # This goes with sines
        k_y_q=np.pi/2*(2*q+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x=np.sin(k_x_i*X_G); 
        cos_k_j_x=np.cos(k_x_j*X_G); 
        sin_k_m_y=np.sin(k_y_m*Y_G); cos_k_m_y=np.cos(k_y_m*Y_G)
        cos_k_q_y=np.cos(k_y_q*Y_G); sin_k_q_y=np.sin(k_y_q*Y_G)
                                
        # Assign the column of Phi_H
        Prime=-(k_y_q*sin_k_m_y*sin_k_q_y-k_y_m*cos_k_m_y*cos_k_q_y)   
        Phi_H_y[:,count]=Prime*sin_k_i_x*cos_k_j_x
        count+=1  

    return Phi_H_y


def Phi_H_2D_Laplacian(X_G,Y_G,n_hb):
    # Get the Laplacian at the points (X_G,Y_G) from n_hb homogeneous spectral
    # basis element. The output is a matrix of side (n_p) x (n_c+n_hb**4)

    # number of points
    n_p = len(X_G)    
       
    # The number of harmonic bases will be:
    n_h = n_hb**4 # number of possible dispositions of the harmonic basis in R2.    
    Lap_H = np.zeros((n_p,n_h))  
    count=0 # Counter that will be used to fill the columns
    # Developer note: the basis is:            
    # sin_k_i_x*cos_k_j_x*sin_k_m_y*sin_k_q_y
    
    # Define the indices, for the possible combination of basis elements
    i_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((3, 1, 2, 0)).ravel()
    j_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 3, 2, 1)).ravel()
    m_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 3, 2)).ravel()
    q_s = np.tile(np.arange(n_hb), (n_hb, n_hb, n_hb, 1)).transpose((0, 1, 2, 3)).ravel()

    for count in range(n_h):
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]           
        k_x_i=2*np.pi*(i+1)/1 # This goes with sines
        k_x_j=np.pi/2*(2*j+1)/1 # This goes with cosines
        k_y_m=2*np.pi*(m+1)/1 # This goes with sines
        k_y_q=np.pi/2*(2*q+1)/1 # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i*X_G); cos_k_i_x = np.cos(k_x_i*X_G) 
        cos_k_j_x = np.cos(k_x_j*X_G); sin_k_j_x = np.sin(k_x_j*X_G)
        sin_k_m_y = np.sin(k_y_m*Y_G); cos_k_m_y = np.cos(k_y_m*Y_G)
        cos_k_q_y = np.cos(k_y_q*Y_G); sin_k_q_y = np.sin(k_y_q*Y_G)
                    
        # Compute the derivatives of the harmonic basis sin_k_i_x
        phi_ijmq_xx = -sin_k_m_y*cos_k_q_y*(2*k_x_i*k_x_j*cos_k_i_x*sin_k_j_x+
                                         (k_x_j**2+k_x_i**2)*sin_k_i_x*cos_k_j_x)
        
        phi_ijmq_yy = -sin_k_i_x*cos_k_j_x*(2*k_y_m*k_y_q*cos_k_m_y*sin_k_q_y+\
                                            (k_y_q**2+k_y_m**2)*sin_k_m_y*cos_k_q_y)
        # Assign the column of the Laplacian
        Lap_H[:,count] = phi_ijmq_xx + phi_ijmq_yy
            
    # # # Here's how to see these        
    # plt.scatter(X_G,Y_G,c=Lap_H[:,1])   
     
    # L=np.hstack((Lap_H,Lap_RBF))
    # fig = plt.figure(figsize = (6, 6), dpi = 100)
    # plt.scatter(X_G, Y_G, c = Lap_H[:,0])
    # ax = plt.gca()
    # ax.set_xticks([])
    # ax.set_yticks([])
    # fig.tight_layout()
    return Lap_H


def Phi_H_3D(X_G,Y_G,Z_G,n_hb):
    # Get the basis matrix at the points (X_G,Y_G) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb**6 # number of possible dispositions of the harmonic basis in R2.    
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
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]
        r = r_s[count]; s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2*np.pi*(i+1) # This goes with sines
        k_x_j = np.pi/2*(2*j+1) # This goes with cosines
        k_y_m = 2*np.pi*(m+1) # This goes with sines
        k_y_q = np.pi/2*(2*q+1) # This goes with cosines
        k_y_r = 2*np.pi*(r+1) # This goes with sines
        k_y_s = np.pi/2*(2*s+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i*X_G) 
        cos_k_j_x = np.cos(k_x_j*X_G) 
        sin_k_m_y = np.sin(k_y_m*Y_G) 
        cos_k_q_y = np.cos(k_y_q*Y_G)  
        sin_k_r_z = np.sin(k_y_r*Z_G) 
        cos_k_s_z = np.cos(k_y_s*Z_G)
                                
        # Assign the column of Phi_H
        Phi_H[:,count] = sin_k_i_x*cos_k_j_x * sin_k_m_y*cos_k_q_y * sin_k_r_z*cos_k_s_z
    return Phi_H

def Phi_H_3D_x(X_G,Y_G,Z_G,n_hb):
    # Get the basis matrix at the points (X_G,Y_G) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb**6 # number of possible dispositions of the harmonic basis in R2.    
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
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]
        r = r_s[count]; s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2*np.pi*(i+1) # This goes with sines
        k_x_j = np.pi/2*(2*j+1) # This goes with cosines
        k_y_m = 2*np.pi*(m+1) # This goes with sines
        k_y_q = np.pi/2*(2*q+1) # This goes with cosines
        k_z_r = 2*np.pi*(r+1) # This goes with sines
        k_z_s = np.pi/2*(2*s+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i*X_G); cos_k_i_x = np.cos(k_x_i*X_G) 
        cos_k_j_x = np.cos(k_x_j*X_G); sin_k_j_x = np.sin(k_x_j*X_G)
        sin_k_m_y = np.sin(k_y_m*Y_G)
        cos_k_q_y = np.cos(k_y_q*Y_G)
        sin_k_r_z = np.sin(k_z_r*Z_G)
        cos_k_s_z = np.cos(k_z_s*Z_G)
                                
        # Assign the column of Phi_H_x
        Prime = -(k_x_j*sin_k_i_x*sin_k_j_x - k_x_i*cos_k_i_x*cos_k_j_x)   
        Phi_H_x[:,count] = Prime * sin_k_m_y*cos_k_q_y * sin_k_r_z*cos_k_s_z
    return Phi_H_x

def Phi_H_3D_y(X_G,Y_G,Z_G,n_hb):
    # Get the basis matrix at the points (X_G,Y_G) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb**6 # number of possible dispositions of the harmonic basis in R2.    
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
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]
        r = r_s[count]; s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2*np.pi*(i+1) # This goes with sines
        k_x_j = np.pi/2*(2*j+1) # This goes with cosines
        k_y_m = 2*np.pi*(m+1) # This goes with sines
        k_y_q = np.pi/2*(2*q+1) # This goes with cosines
        k_z_r = 2*np.pi*(r+1) # This goes with sines
        k_z_s = np.pi/2*(2*s+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i*X_G)
        cos_k_j_x = np.cos(k_x_j*X_G)
        sin_k_m_y = np.sin(k_y_m*Y_G); cos_k_m_y = np.cos(k_y_m*Y_G)
        cos_k_q_y = np.cos(k_y_q*Y_G); sin_k_q_y = np.sin(k_y_q*Y_G) 
        sin_k_r_z = np.sin(k_z_r*Z_G)
        cos_k_s_z = np.cos(k_z_s*Z_G)
                                
        # Assign the column of Phi_H_y
        Prime = -(k_y_m*sin_k_m_y*sin_k_q_y - k_y_q*cos_k_m_y*cos_k_q_y)   
        Phi_H_y[:,count] = Prime * sin_k_i_x*cos_k_j_x * sin_k_r_z*cos_k_s_z
    return Phi_H_y

def Phi_H_3D_z(X_G,Y_G,Z_G,n_hb):
    # Get the basis matrix at the points (X_G,Y_G) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb**6 # number of possible dispositions of the harmonic basis in R2.    
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
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]
        r = r_s[count]; s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2*np.pi*(i+1) # This goes with sines
        k_x_j = np.pi/2*(2*j+1) # This goes with cosines
        k_y_m = 2*np.pi*(m+1) # This goes with sines
        k_y_q = np.pi/2*(2*q+1) # This goes with cosines
        k_z_r = 2*np.pi*(r+1) # This goes with sines
        k_z_s = np.pi/2*(2*s+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i*X_G)
        cos_k_j_x = np.cos(k_x_j*X_G)
        sin_k_m_y = np.sin(k_y_m*Y_G)
        cos_k_q_y = np.cos(k_y_q*Y_G)
        sin_k_r_z = np.sin(k_z_r*Z_G); cos_k_r_z = np.cos(k_z_r*Z_G)
        cos_k_s_z = np.cos(k_z_s*Z_G); sin_k_s_z = np.sin(k_z_s*Z_G)
                      
        # Assign the column of Phi_H_z
        Prime = -(k_z_s*sin_k_r_z*sin_k_s_z - k_z_r*cos_k_r_z*cos_k_s_z)   
        Phi_H_z[:,count] = Prime * sin_k_i_x*cos_k_j_x * sin_k_m_y*cos_k_q_y
    return Phi_H_z

def Phi_H_3D_Laplacian(X_G,Y_G,Z_G,n_hb):
    # Get the basis matrix at the points (X_G,Y_G) from n_hb homogeneous 
    # spectral basis element.       
    # The output is a matrix of side (n_p) x (n_hb**4)
    # Get the number of points
    n_p = len(X_G)
    # This is the contribution of the harmonic part (sines and cosines)
    # The number of harmonic bases will be:
    n_h = n_hb**6 # number of possible dispositions of the harmonic basis in R2.    
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
        i = i_s[count]; j = j_s[count]
        m = m_s[count]; q = q_s[count]
        r = r_s[count]; s = s_s[count]
        # print(i, j, l, m)
        k_x_i = 2*np.pi*(i+1) # This goes with sines
        k_x_j = np.pi/2*(2*j+1) # This goes with cosines
        k_y_m = 2*np.pi*(m+1) # This goes with sines
        k_y_q = np.pi/2*(2*q+1) # This goes with cosines
        k_z_r = 2*np.pi*(r+1) # This goes with sines
        k_z_s = np.pi/2*(2*s+1) # This goes with cosines
        # To take the differentiation, we use automatic diff style:
        sin_k_i_x = np.sin(k_x_i*X_G); cos_k_i_x = np.cos(k_x_i*X_G) 
        cos_k_j_x = np.cos(k_x_j*X_G); sin_k_j_x = np.sin(k_x_j*X_G)
        sin_k_m_y = np.sin(k_y_m*Y_G); cos_k_m_y = np.cos(k_y_m*Y_G)
        cos_k_q_y = np.cos(k_y_q*Y_G); sin_k_q_y = np.sin(k_y_q*Y_G) 
        sin_k_r_z = np.sin(k_z_r*Z_G); cos_k_r_z = np.cos(k_z_r*Z_G)
        cos_k_s_z = np.cos(k_z_s*Z_G); sin_k_s_z = np.sin(k_z_s*Z_G)
                                
        # Compute the derivatives of the harmonic basis sin_k_i_x
        phi_ijmqrs_xx = -sin_k_m_y*cos_k_q_y * sin_k_r_z*cos_k_s_z * \
            (2*k_x_i*k_x_j*cos_k_i_x*sin_k_j_x + (k_x_j**2+k_x_i**2)*sin_k_i_x*cos_k_j_x)
        
        phi_ijmqrs_yy = -sin_k_i_x*cos_k_j_x * sin_k_r_z*cos_k_s_z * \
            (2*k_y_m*k_y_q*cos_k_m_y*sin_k_q_y + (k_y_q**2+k_y_m**2)*sin_k_m_y*cos_k_q_y)
        
        phi_ijmqrs_zz = -sin_k_i_x*cos_k_j_x * sin_k_m_y*cos_k_q_y * \
            (2*k_z_r*k_z_s*cos_k_r_z*sin_k_s_z + (k_z_r**2+k_z_s**2)*sin_k_r_z*cos_k_s_z)            
            
        # Assign the column of the Laplacian
        Lap_H[:,count] = phi_ijmqrs_xx + phi_ijmqrs_yy + phi_ijmqrs_zz
    return Lap_H


def add_constraint_collocations_2D(X_constr, Y_constr, X_C, Y_C, r_mM, eps_l, basis):
    """
    This function adds collocation points where constraints are set in 2D.
    
    ----------------------------------------------------------------------------------------------------------------
    Parameters
    ----------
    :param X_constr: np.ndarray
        X coordinates of the constraints
    :param Y_constr: np.ndarray
        Y coordinates of the constraints
    :param X_C: np.ndarray
        X coordinates of the collocation points
    :param Y_C: np.ndarray
        Y coordinates of the collocation points
    :param r_mM: list
        Minimum and maximum radius of the RBFs
    :param eps_l: float
        Value of the RBF at its closest neighbor
    :param basis: str
        Type of basis function, must be c4 or Gaussian
    """   
    # Get the number of constraints
    n_constr = X_constr.shape[0]
    # Initialize an empty array for the shape parameters
    c_ks = np.zeros(n_constr)
    
    # Check the basis
    if basis == 'gauss': # Gaussians
        # Set the max and min values of c_k  
        c_min = 1 / (2*r_mM[1]) * np.sqrt(np.log(2))
        c_max = 1 / (2*r_mM[0]) * np.sqrt(np.log(2))
        # Loop over all constraints
        for k in range(n_constr):
            # Get the distance to all collocation points
            dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 + (Y_C - Y_constr[k])**2)
            # Get the distance to all constraints, except for itself
            dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
                                     (np.delete(Y_constr, k) - Y_constr[k])**2)
            # Set the max and min values of c_k 
            c_k = np.sqrt(-np.log(eps_l)) / np.concatenate((dist_to_colloc, dist_to_constr))
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # get the maximum value in the case of the Gaussian
            c_ks[k] = np.max(c_k)
        # for plotting purposes, we store also the diameters             
        d_k=1/c_ks*np.sqrt(np.log(2))      
        
    elif basis == 'c4': # C4
        c_min = 2*r_mM[0] / np.sqrt(1 - 0.5**0.2)
        c_max = 2*r_mM[1] / np.sqrt(1 - 0.5**0.2)
        for k in range(n_constr):
            # Get the distance to all collocation points
            dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 + (Y_C - Y_constr[k])**2)
            # Get the distance to all constraints, except for itself
            dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
                                     (np.delete(Y_constr, k) - Y_constr[k])**2)
            # Set the max and min values of c_k 
            c_k = np.concatenate((dist_to_colloc, dist_to_constr)) / np.sqrt(1 - eps_l**0.2)
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # get the minimum value in the case of the c4
            c_ks[k] = np.min(c_k)
        # for plotting purposes, we store also the diameters
        d_k = c_ks * np.sqrt(1 - 0.5**0.2)
    
    return c_ks*2, d_k

def add_constraint_collocations_3D(X_constr, Y_constr, Z_constr, X_C, Y_C, Z_C, r_mM, eps_l, basis):
    """
    This function adds collocation points where constraints are set in 3D.
    
    ----------------------------------------------------------------------------------------------------------------
    Parameters
    ----------
    :param X_constr: np.ndarray
        X coordinates of the constraints
    :param Y_constr: np.ndarray
        Y coordinates of the constraints
    :param Z_constr: np.ndarray
        Z coordinates of the constraints
    :param X_C: np.ndarray
        X coordinates of the collocation points
    :param Y_C: np.ndarray
        Y coordinates of the collocation points
    :param Z_C: np.ndarray
        Z coordinates of the collocation points
    :param r_mM: list
        Minimum and maximum radius of the RBFs
    :param eps_l: float
        Value of the RBF at its closest neighbor
    :param basis: str
        Type of basis function, must be c4 or Gaussian
    """   
    # Get the number of constraints
    n_constr = X_constr.shape[0]
    # Initialize an empty array for the shape parameters
    c_ks = np.zeros(n_constr)
    
    # Check the basis
    if basis == 'gauss': # Gaussians
        # Set the max and min values of c_k  
        c_min = 1 / (2*r_mM[1]) * np.sqrt(np.log(2))
        c_max = 1 / (2*r_mM[0]) * np.sqrt(np.log(2))
        # Loop over all constraints
        for k in range(n_constr):
            # Get the distance to all collocation points
            dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 +\
                                     (Y_C - Y_constr[k])**2 +\
                                     (Z_C - Z_constr[k])**2)
            # Get the distance to all constraints, except for itself
            dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
                                     (np.delete(Y_constr, k) - Y_constr[k])**2+\
                                     (np.delete(Z_constr, k) - Z_constr[k])**2)
            # Set the max and min values of c_k 
            c_k = np.sqrt(-np.log(eps_l)) / np.concatenate((dist_to_colloc, dist_to_constr))
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # get the maximum value in the case of the Gaussian
            c_ks[k] = np.max(c_k)
        # for plotting purposes, we store also the diameters             
        d_k=1/c_ks*np.sqrt(np.log(2))      
        
    elif basis == 'c4': # C4
        c_min = 2*r_mM[0] / np.sqrt(1 - 0.5**0.2)
        c_max = 2*r_mM[1] / np.sqrt(1 - 0.5**0.2)
        for k in range(n_constr):
            # Get the distance to all collocation points
            dist_to_colloc = np.sqrt((X_C - X_constr[k])**2 +\
                                     (Y_C - Y_constr[k])**2 +\
                                     (Z_C - Z_constr[k])**2)
            # Get the distance to all constraints, except for itself
            dist_to_constr = np.sqrt((np.delete(X_constr, k) - X_constr[k])**2+\
                                     (np.delete(Y_constr, k) - Y_constr[k])**2+\
                                     (np.delete(Z_constr, k) - Z_constr[k])**2)
            # Set the max and min values of c_k 
            c_k = np.concatenate((dist_to_colloc, dist_to_constr)) / np.sqrt(1 - eps_l**0.2)
            # crop to the minimum and maximum value
            c_k[c_k < c_min] = c_min
            c_k[c_k > c_max] = c_max
            # get the minimum value in the case of the c4
            c_ks[k] = np.min(c_k)
        # for plotting purposes, we store also the diameters
        d_k = c_ks * np.sqrt(1 - 0.5**0.2)
    
    return c_ks, d_k

# =============================================================================
# PUM Functions
# =============================================================================

def Psi_Wendland_2D(X_G, Y_G, X_Circ, Y_Circ, radius):
    d_norm = np.sqrt((X_Circ - X_G)**2 + (Y_Circ - Y_G)**2) / radius
    Psi = (4*d_norm+1) * (1 - d_norm)**4
    Psi[d_norm > 1] = 0
    return Psi
    