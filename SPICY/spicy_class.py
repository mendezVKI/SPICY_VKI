# -*- coding: utf-8 -*-
"""
Latest update on Thu Jan 12 17:56:06 2023

@author: mendez, ratz, sperotto
"""

# Test

import numpy as np

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
        Definition of the class spicy
             
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
    
    


# #%% 1. Clustering (this does not depend on the model, but only on the dimension).
# def clustering(self,N,cap,mincluster=[False],el=np.exp(-0.5**2/2),collocation_augmentation=0):
#       """
#        This function defines the collocation of a set of RBFs using the multi-level clustering.             
        
#         ----------------------------------------------------------------------------------------------------------------
#         Parameters
#         ----------
#         :param constraint: tuple,
#                     In the first two or three cell depending on the dimensionality contains a list containing again the arrays of every
#                     boundary or constrained points. The last cell is a list containing a list of 2 or 3 list depending on the dimensionality contains a list containing again the arrays of every
#                     boundary or constrained points. Sincerely don't know how to explain this check the test case'
                    
        
#         ----------------------------------------------------------------------------------------------------------------
#       """
    
#     return



# def plot_RBFs:
    
    
#     return


# #%% 2. Constraints (this depends on everything)
# def constraints:
    
#     return

# #%% 3. Assembly A, B, C(this depends on everything)

# def Assembly:
    
#     return








    
    

