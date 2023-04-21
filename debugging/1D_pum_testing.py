# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 11:52:32 2022

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh

# Configuration for plots
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)

#%% Problem to Fit
x1 = np.linspace(0, 4.3, 200, endpoint=True)
x2 = np.linspace(4.3, 10, 200, endpoint=True)[1:]
x=np.concatenate((x1,x2))
# Create the deterministic part
y_clean= 3*x+(x/100)**3+4*np.sin(3/2*np.pi*x)
# Add (a seeded) stochastic part
np.random.seed(0)
y=y_clean+0.5*np.random.randn(len(x))
# Introduce some outliers in x=2 and x=8
G1=10*np.exp(-(x-2)**2/0.005)*np.random.randn(len(x))
G2=15*np.exp(-(x-8)**2/0.005)*np.random.randn(len(x))
y_final=y+G1+G2
y_final = y
# y_final = y_clean

#%% Create Feature matrix for the Sigmoid
# Define the number of bases
n_b = 100

# Perform the feature scaling

xmin = x.min(); xmax = x.max()
ymin = y_clean.min(); ymax = y_clean.max()

x_prime = (x - xmin) / (xmax - xmin)
y_prime = (y_final - ymin) / (ymax - ymin)
y_clean = (y_clean - ymin) / (ymax - ymin)

# Define grid of collocation points
x_b=np.linspace(0, 1, n_b)

#%% 3 C4 Basis

def C4_Compact_RBF(x,x_r=0,c_r=0.1):
    d=x-x_r # Get distance
    phi_r=(1+d/c_r)**5*(1-d/c_r)**5
    phi_r[np.abs(d)>c_r]=0
    return phi_r

def PHI_C4(x_in, x_b, c_r=0.1):
    n_x=np.size(x_in); n_b=len(x_b)
    Phi_X=np.zeros((n_x,n_b)) # Initialize Basis Matrix on x
    # Add a constant and a linear term
    Phi_X[:,0]=x_in 
    for j in range(0,n_b): # Loop to prepare the basis matrices (inefficient)
        Phi_X[:,j]=C4_Compact_RBF(x_in,x_r=x_b[j],c_r=c_r)  # Prepare all the terms in the basis 
    return Phi_X


# %%
# Utility function for comparing conditioning numbers
import math
def OOM(number):
    return math.floor(math.log(number, 10))

def Psi_Wendland(X_G, X_Circ, radius):
    d_norm = np.sqrt((X_Circ - X_G)**2) / radius
    Psi = (4*d_norm+1) * (1 - d_norm)**4
    Psi[d_norm > 1] = 0
    return Psi

def RBF_global(x_prime, y_prime, x_b, c_r, K_cond):
    # Compute the Matrix Phi on ALL grid points and ALL collocation points
    Phi = PHI_C4(x_prime, x_b, c_r=c_r)
    # Compute A
    A = Phi.T@Phi
    # Compute b_1
    b_1 = Phi.T@y_prime
    # Check the conditioning
    print('Conditioning number of global system: 10^' + str(OOM(np.linalg.cond(A))))
    
    # Regularize the matrix
    lambda_A = eigsh(A, 1, return_eigenvectors=False) # Largest eigenvalue
    alpha = lambda_A / K_cond
    A = A + alpha * np.eye(A.shape[0])
    # Compute the weights
    w_C4 = np.linalg.inv(A).dot(b_1)
    # We test on the same points
    y_global=Phi.dot(w_C4)
    
    # Show Matrix A
    plt.figure(figsize = (5, 5), dpi = 100)
    plt.imshow(A)
    plt.title('Structure of one A for RBF global')
    
    return y_global

def RBF_pum_hybrid(x_prime, y_prime, x_b, c_r, K_cond, X_centers, radius):
    """
    This is the hybrid between local and global PUM that we derived together
    """
    # get the number of points
    n_p = x_prime.shape[0]
    # get the number of patches
    n_patches = X_centers.shape[0]

    # Compute the Psi-weights of the grid points
    Matrix_Wendland_X = np.zeros((n_p, n_patches))
    for i in range(n_patches):
        x_circ_center = X_centers[i]
        # Compute the local Wendland functions in each column
        Matrix_Wendland_X[:,i] = Psi_Wendland(x_prime, x_circ_center, radius)
    # Compute the Psi-matrix for each grid point
    Matrix_Psi_X = Matrix_Wendland_X / np.sum(Matrix_Wendland_X, axis = 1)[:, np.newaxis]

    # initialize an empty array for the weights of the global system
    weights = np.zeros((n_b, n_patches))
    
    # loop over all of the patches
    for i in range(n_patches):
        # get the data points in the patch
        in_patch_g = (x_prime - X_centers[i])**2 < radius**2
        x_g_j = x_prime[in_patch_g]
        y_g_j = y_prime[in_patch_g]      

        # we use all collocation points (globally)
        in_patch_c = np.ones(x_b.shape, dtype = bool)
        x_b_j = x_b[in_patch_c]
          
        # Compute the phi matrix of the points in the patch                     
        Matrix_Phi_on_x_g_j = PHI_C4(x_g_j, x_b_j)
        # Psi-weigh them according to the Psi-weights associated to the grid points in the patch
        Matrix_Phi_on_x_g_j_weighted = np.diag(Matrix_Psi_X[in_patch_g,i])@Matrix_Phi_on_x_g_j
        
        # Get the indices of the RBFs which are contributing something. Their columns
        # are non zero, so they can be easily filtered. This step can be done 
        # more efficiently, but for now it is a quick solution
        indices_contributing_rbfs = np.abs(Matrix_Phi_on_x_g_j.sum(axis = 0)) != 0
        # Extract the reduced, weighted phi matrix
        Matrix_Phi_on_x_g_j_weighted_reduced = Matrix_Phi_on_x_g_j_weighted[:,indices_contributing_rbfs]
        # Compute Phi.T @ Phi of this reduced matrix
        A = Matrix_Phi_on_x_g_j_weighted_reduced.T@Matrix_Phi_on_x_g_j_weighted_reduced
        # Psi-weigh the target values according to the Psi-weights associated to the grid points in the patch
        y_weighted = np.diag(Matrix_Psi_X[in_patch_g,i])@y_g_j
        
        # regularize the matrix (it is quite badly conditioned)
        print('Conditioning number for PUM hybrid of local patch ' + str(i) + ': 10^' + str(OOM(np.linalg.cond(A))))
        lambda_A = eigsh(A, 1, return_eigenvectors=False) # Largest eigenvalue
        alpha = lambda_A / K_cond
        A = A + alpha * np.eye(A.shape[0])
        
        # compute the Psi-weights (direct method)
        w = np.linalg.inv(A).dot(Matrix_Phi_on_x_g_j_weighted_reduced.T).dot(y_weighted)   
        # add the Psi-weights of the local target values to the columns of the global weight matrix
        weights[indices_contributing_rbfs,i] = w
    
    # Compute the Psi-weights of the collocation points
    Matrix_Wendland_X_b = np.zeros((n_b, n_patches))
    for i in range(n_patches):
        x_circ_center = X_centers[i]
        # Compute the local Wendland functions in each column
        Matrix_Wendland_X_b[:,i] = Psi_Wendland(x_b, x_circ_center, radius)
    # Compute the Psi-matrix for each collocation point
    Matrix_Psi_X_b = Matrix_Wendland_X_b / np.sum(Matrix_Wendland_X_b, axis = 1)[:, np.newaxis]

    # Compute the global Phi matrix
    Phi_global = PHI_C4(x_prime, x_b)
    # Project the weights with the Psi function
    weights_weighted = np.multiply(Matrix_Psi_X_b, weights).sum(axis = 1)
    # and compute the global prediction (one matrix-vector product)
    y_pum_hybrid = Phi_global@weights_weighted
    
    # Show Matrix A
    plt.figure(figsize = (5, 5), dpi = 100)
    plt.imshow(A)
    plt.title('Structure of one A for PUM hybrid')
    
    return y_pum_hybrid

def RBF_pum_local(x_prime, y_prime, x_b, c_r, K_cond, X_centers, radius):
    """
    This is the local PUM that is done in interpolation
    """
    # get the number of points
    n_p = x_prime.shape[0]
    # get the number of patches
    n_patches = X_centers.shape[0]

    # Compute the Psi-weights of the grid points
    Matrix_Wendland_X = np.zeros((n_p, n_patches))
    for i in range(n_patches):
        x_circ_center = X_centers[i]
        # Compute the local Wendland functions in each column
        Matrix_Wendland_X[:,i] = Psi_Wendland(x_prime, x_circ_center, radius)
    # Compute the Psi-matrix for each grid point
    Matrix_Psi_X = Matrix_Wendland_X / np.sum(Matrix_Wendland_X, axis = 1)[:, np.newaxis]

    # # initialize an empty array for the global solution
    y_RBF_pum_local = np.zeros((n_p))

    for i in range(n_patches):
        # get the data points in the patch
        in_patch_g = (x_prime - X_centers[i])**2 < radius**2
        x_g_j = x_prime[in_patch_g]
        y_g_j = y_prime[in_patch_g]      

        # we use the collocation points in the patch, otherwise, we recover the
        # global system again, just with the different weights. This means, the 
        # resulting, dense matrix would be the same
        in_patch_c = (x_b - X_centers[i])**2 < radius**2 
        x_b_j = x_b[in_patch_c]
          
        # Compute the phi matrix of the points in the patch                     
        Matrix_Phi_on_x_g_j = PHI_C4(x_g_j, x_b_j)
        # Psi-weigh them according to the Psi-weights associated to the grid points in the patch
        Matrix_Phi_on_x_g_j_weighted = np.diag(Matrix_Psi_X[in_patch_g,i])@Matrix_Phi_on_x_g_j
        
        # Get the indices of the RBFs which are contributing something. Their columns
        # are non zero, so they can be easily filtered
        indices_contributing_rbfs = np.abs(Matrix_Phi_on_x_g_j.sum(axis = 0)) != 0
        # Extract the reduced, weighted phi matrix
        Matrix_Phi_on_x_g_j_weighted_reduced = Matrix_Phi_on_x_g_j_weighted[:,indices_contributing_rbfs]
        # Compute Phi.T @ Phi of this reduced matrix
        A = Matrix_Phi_on_x_g_j_weighted_reduced.T@Matrix_Phi_on_x_g_j_weighted_reduced
        # Psi-weigh the target values according to the Psi-weights associated to the grid points in the patch
        y_weighted = np.diag(Matrix_Psi_X[in_patch_g,i])@y_g_j
        
        # regularize the matrix (it is quite badly conditioned)
        print('Conditioning number for PUM Local of local patch ' + str(i) + ': 10^' + str(OOM(np.linalg.cond(A))))
        lambda_A = eigsh(A, 1, return_eigenvectors=False) # Largest eigenvalue
        alpha = lambda_A / K_cond
        A = A + alpha * np.eye(A.shape[0])
        
        # compute the Psi-weights (direct method)
        w = np.linalg.inv(A).dot(Matrix_Phi_on_x_g_j_weighted_reduced.T).dot(y_weighted)   
        # add the Psi-weights of the local target values to the columns of the global weight matrix
        y_RBF_pum_local[in_patch_g] += Matrix_Phi_on_x_g_j_weighted_reduced@w
    
    # Show Matrix A
    plt.figure(figsize = (5, 5), dpi = 100)
    plt.imshow(A)
    plt.title('Structure of one A for PUM local')
    
    return y_RBF_pum_local

def RBF_pum_global(x_prime, y_prime, x_b, c_r, K_cond, X_centers, radius):
    """
    This is the local PUM that is done in interpolation
    """
    # get the number of points
    n_p = x_prime.shape[0]
    # get the number of patches
    n_patches = X_centers.shape[0]

    # Compute the Psi-weights of the grid points
    Matrix_Wendland_X = np.zeros((n_p, n_patches))
    for i in range(n_patches):
        x_circ_center = X_centers[i]
        # Compute the local Wendland functions in each column
        Matrix_Wendland_X[:,i] = Psi_Wendland(x_prime, x_circ_center, radius)
    # Compute the Psi-matrix for each grid point
    Matrix_Psi_X = Matrix_Wendland_X / np.sum(Matrix_Wendland_X, axis = 1)[:, np.newaxis]
    
    # Allocate the global Phi Matrix
    Phi = np.zeros((n_p, x_b.shape[0])).ravel()
    
    for i in range(n_patches):
        # get the data points in the patch
        in_patch_g = (x_prime - X_centers[i])**2 < radius**2
        x_g_j = x_prime[in_patch_g]    
        
        # we only take the basis functions in the patch, otherwise, we recover the global (dense) system
        in_patch_c = (x_b - X_centers[i])**2 < radius**2 
        x_b_j = x_b[in_patch_c]
          
        # Compute the phi matrix of the points in the patch                     
        Matrix_Phi_on_x_g_j = PHI_C4(x_g_j, x_b_j)
        # Psi-weigh them according to the Psi-weights associated to the grid points in the patch
        Matrix_Phi_on_x_g_j_weighted = np.diag(Matrix_Psi_X[in_patch_g,i])@Matrix_Phi_on_x_g_j
        
        in_circle_c_gridded_with_g, in_patch_g_gridded_with_c = np.meshgrid(in_patch_c, in_patch_g)
        index_mapping_local_to_global = np.multiply(in_circle_c_gridded_with_g, in_patch_g_gridded_with_c)
        Phi[index_mapping_local_to_global.ravel()] += Matrix_Phi_on_x_g_j_weighted.ravel()
    
    # Reshape Phi into a 2D matrix
    Phi = Phi.reshape(n_p, x_b.shape[0])
    
    # Assemble A and b_1
    A = Phi.T@Phi
    b_1 = Phi.T@y_prime
    # Check the conditioning
    print('Conditioning number of PUM global system: 10^' + str(OOM(np.linalg.cond(A))))
    
    
    # Regularize the global system
    lambda_A = eigsh(A, 1, return_eigenvectors=False) # Largest eigenvalue
    alpha = lambda_A / K_cond
    A = A + alpha * np.eye(A.shape[0])
    # Train Model
    w_C4 = np.linalg.inv(A)@b_1
    # We test on the same points
    y_pum_global = Phi.dot(w_C4)
    
    # Show Matrix A
    plt.figure(figsize = (5, 5), dpi = 100)
    plt.imshow(A)
    plt.title('Structure of A for PUM global')
    
    # Show Matrix Phi
    plt.figure(figsize = (5, 5), dpi = 100)
    plt.imshow(Phi)
    plt.title('Structure of Phi for PUM global')
    
    return y_pum_global

# Globally set shape parameter for all methods
c_r = 0.1
# Condition number for each regression problem (local or global). In particular, 
# the hybrid method suffers when it is too large. For 1e8, it is the only method 
# which crashes. The others are very comparable. If we increase the conditioning 
# number to 1e4, the hybrid method still performs worse in the regions of overlap
# while the other three are still very comparable. For 1e3, all four methods are 
# comparable, but the error between the global method and the other three is the 
# largest for this case.
K_cond = 1e3

# Location of the X centers
X_centers = np.linspace(0.1, 0.9, 5)
# Number of patches
n_patches = X_centers.shape[0]
# Radius of the patches
radius = 0.15

# Perform the different RBF regressions
y_RBF_global = RBF_global(x_prime, y_prime, x_b, c_r, K_cond)
y_RBF_pum_hybrid = RBF_pum_hybrid(x_prime, y_prime, x_b, c_r, K_cond, X_centers, radius)
y_RBF_pum_local = RBF_pum_local(x_prime, y_prime, x_b, c_r, K_cond, X_centers, radius)
y_RBF_pum_global = RBF_pum_global(x_prime, y_prime, x_b, c_r, K_cond, X_centers, radius)

# Global comparison of the predictions
fig, ax = plt.subplots(figsize = (7, 5), dpi = 100) 
ax.scatter(x_prime, y_prime, c='white', marker='o',edgecolor='black',
            s=5,label='Data')
ax.plot(x_prime, y_RBF_global, label = 'RBF Global')
ax.plot(x_prime, y_RBF_pum_hybrid, label = 'RBF PUM hybrid')
ax.plot(x_prime, y_RBF_pum_local, label = 'RBF PUM local')
ax.plot(x_prime, y_RBF_pum_global, label = 'RBF PUM global')
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16) 
ax.legend()

# Print the errors
error_global = np.linalg.norm(y_RBF_global - y_clean) / np.linalg.norm(y_clean)
print('Error with RBF Global:     {0:.3f} %'.format(error_global*100))

error_pum = np.linalg.norm(y_RBF_pum_hybrid - y_clean) / np.linalg.norm(y_clean)
print('Error with RBF PUM hybrid: {0:.3f} %'.format(error_pum*100))

error_pum_local = np.linalg.norm(y_RBF_pum_local - y_clean) / np.linalg.norm(y_clean)
print('Error with RBF PUM local:  {0:.3f} %'.format(error_pum_local*100))

error_pum_global = np.linalg.norm(y_RBF_pum_global - y_clean) / np.linalg.norm(y_clean)
print('Error with RBF PUM global: {0:.3f} %'.format(error_pum_global*100))

# Show the errors between the prediction and the ground truth
fig, ax = plt.subplots(figsize = (7, 5), dpi = 100)
ax.plot(x_prime, y_RBF_global - y_clean, label = 'RBF Global')
ax.plot(x_prime, y_RBF_pum_hybrid - y_clean, label = 'RBF PUM hybrid')
ax.plot(x_prime, y_RBF_pum_local - y_clean, label = 'RBF PUM local')
ax.plot(x_prime, y_RBF_pum_global - y_clean, label = 'RBF PUM global')
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('error y',fontsize=16)
ax.legend()




# Thoughts during coding
"""
In the pum_global approach, only the local basis functions should play a role.
Otherwise, we recover the general global approach which is not what we want.

For pum_local and pum_hybrid, it only affects the computational cost for forming
the matrix A. However, if we then reproject the weights and add them to the
global solution, only the ones in the patch are actually added
"""