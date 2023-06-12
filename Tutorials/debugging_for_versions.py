# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:43:39 2023

@author: manue
"""
import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
import os
import sys
sys.path.append('..' + os.sep + 'spicy_vki' + os.sep + 'spicy')
from spicy_class import spicy

# This is for plot customization
fontsize = 12
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['font.size'] = fontsize

# Fix random seed to ensure reproducibility
np.random.seed(42)

# Properties of the domain and flow
R = 0.05 # m
H = 0.41 # m
L = 1.1 # m
mu = 2e-2 # Pa s
rho = 1 # kg/m^3

# Load the matlab data from the ansys solution
data = scipy.io.loadmat('FluentSol.mat')
# Extract the x, y values
X = data['x'].reshape(-1); Y = data['y'].reshape(-1) 
# Extract the velocities and pressure
U = data['vx'].reshape(-1); V = data['vy'].reshape(-1); P = data['p'].reshape(-1)

# Here, we remove the points at the inlet and at the wall, as they are given by the constraints
inlet_and_wall_remover = np.invert(np.logical_or(np.logical_and(U==0, V==0), X==0))
# Remove the points
X_p = X[inlet_and_wall_remover]; Y_p = Y[inlet_and_wall_remover]; P_p = P[inlet_and_wall_remover]
U_p = U[inlet_and_wall_remover]; V_p = V[inlet_and_wall_remover]


# n_p = 5000
# random_points_indices = np.random.randint(low=0, high=len(X_p), size=n_p)
# # Select the data points
# X_p = X_p[random_points_indices]
# Y_p = Y_p[random_points_indices]
# P_p = P_p[random_points_indices]
# U_p = U_p[random_points_indices]
# V_p = V_p[random_points_indices]

# A testing data, we take the ones that are not in this list.
from sklearn.model_selection import train_test_split
indices_train, indices_test=train_test_split(np.arange(0,len(X_p),1),test_size=0.2)

n_p = len(indices_train) # This is the number of data points that will be used for training

# Select the data points that will be used for  training 
X = X_p[indices_train]; Y = Y_p[indices_train]; P = P_p[indices_train]
U = U_p[indices_train]; V = V_p[indices_train]

# Add 0.3 noise to the velocity field
q = 0.05
U_noise = U * (1 + q * np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q * np.random.uniform(-1, 1, size = V.shape))

# # Let's have a look at the velocity data:
# fig=plt.figure(11)
# plt.scatter(X,Y,c=np.sqrt(U**2+V**2))
# plt.gca().set_aspect(1)

## Boundary and Constraint Definitions
n_c = 50

# Left boundary (index: 1)
X_Div1 = np.zeros(n_c)
Y_Div1 = np.linspace(0, H, n_c)
U_Dir1 = 4*(H-Y_Div1)*Y_Div1/H**2
V_Dir1 = np.zeros(X_Div1.shape)
# Bottom boundary (index: 2)
X_Div2 = np.linspace(0, L, n_c)
Y_Div2 = np.zeros(n_c)
U_Dir2 = np.zeros(X_Div2.shape)
V_Dir2 = np.zeros(X_Div2.shape)
# Right boundary (index: 3)
X_Div3 = np.ones(n_c-2)*L
Y_Div3 = np.linspace(0, H, n_c)[1:-1]
# Top boundary (index: 4)
X_Div4 = np.linspace(0, L, n_c)
Y_Div4 = np.ones(n_c)*H
U_Dir4 = np.zeros(X_Div4.shape)
V_Dir4 = np.zeros(X_Div4.shape)
# Cylinder boundary (index: 4)
alphaT = np.linspace(0, 2*np.pi, 20*n_c, endpoint = False)
X_Div5 = 0.2+R*np.cos(alphaT)
Y_Div5 = 0.2+R*np.sin(alphaT)
U_Dir5 = np.zeros(X_Div5.shape)
V_Dir5 = np.zeros(X_Div5.shape)

# We assemble the velocity constraints for Dirichlet
X_Dir = np.concatenate((X_Div1, X_Div2, X_Div4, X_Div5))
Y_Dir = np.concatenate((Y_Div1, Y_Div2, Y_Div4, Y_Div5))
U_Dir = np.concatenate((U_Dir1, U_Dir2, U_Dir4, U_Dir5))
V_Dir = np.concatenate((V_Dir1, V_Dir2, V_Dir4, V_Dir5))
# and Divergence-free flow
X_Div = np.concatenate((X_Div1, X_Div2, X_Div3, X_Div4, X_Div5))
Y_Div = np.concatenate((Y_Div1, Y_Div2, Y_Div3, Y_Div4, Y_Div5))

# We remove the duplicates in the Dirchlet 
_, valid_idcs = np.unique(np.column_stack((X_Div, Y_Div)), return_index = True, axis = 0)
X_Div = X_Div[valid_idcs]
Y_Div = Y_Div[valid_idcs]
DIV = [X_Div, Y_Div]

# and Divergence-free conditions
_, valid_idcs = np.unique(np.column_stack((X_Dir, Y_Dir)), return_index = True, axis = 0)
X_Dir = X_Dir[valid_idcs]
Y_Dir = Y_Dir[valid_idcs]
U_Dir = U_Dir[valid_idcs]
V_Dir = V_Dir[valid_idcs]
DIR = [X_Dir, Y_Dir, U_Dir, V_Dir]

from shapely import geometry

# Definition of Box 1
p1 = geometry.Point(0.1,0.1); p2 = geometry.Point(0.3,0.1)
p3 = geometry.Point(0.3,0.3); p4 = geometry.Point(0.1,0.3)

pointList = [p1, p2, p3, p4]
poly1 = geometry.Polygon([i for i in pointList])  # BOX one

# Definition of Box 2 

# Define also a second box
p1 = geometry.Point(0.0,0); p2 = geometry.Point(0.5,0)
p3 = geometry.Point(0.5,0.45); p4 = geometry.Point(0.0,0.45)

pointList = [p1, p2, p3, p4]
poly2 = geometry.Polygon([i for i in pointList])

# # Plot the boxes with the velocity field
# fig=plt.figure()
# plt.quiver(X,Y,U,V)
# plt.gca().set_aspect(1)
# plt.plot(*poly2.exterior.xy,'b') # in case you want to see them for info
# plt.plot(*poly1.exterior.xy,'r') # in case you want to see them for info

# Prepare the SPICY object
SP_vel = spicy([U_noise,V_noise], [X,Y], basis='gauss')
# Clustering 
SP_vel.clustering([3,6,50,200], Areas=[poly1,poly2,[],[]], r_mM=[0.015,0.3], eps_l=0.87)
# Introduce the Constraints
SP_vel.vector_constraints(DIR=DIR, DIV=DIV, extra_RBF=False)
# Plot the RBF cluster
SP_vel.plot_RBFs(l=0)
SP_vel.plot_RBFs(l=1)
SP_vel.plot_RBFs(l=2)
SP_vel.plot_RBFs(l=3)

# Assembly the system
SP_vel.Assembly_Regression(alpha_div=1) 
# Solve the system
SP_vel.Solve(K_cond=1e12)

# Get the solution
U_calc,V_calc=SP_vel.Get_Sol([X,Y])

# Magnitude of the RBF solution
U_magn_calc = np.sqrt(U_calc**2 + V_calc**2)
# Compute the magnitude of the analytical solution
U_magn = np.sqrt(U**2 + V**2)
# Compute the error in the magnitude
error_magn = np.linalg.norm(U_magn_calc - U_magn) / np.linalg.norm(U_magn)
# Error in u
error_u = np.linalg.norm(U_calc - U) / np.linalg.norm(U)
# Error in v
error_v = np.linalg.norm(V_calc - V) / np.linalg.norm(V)

print('------- In-sample error results ---------')
print('Total velocity error: {0:.3f}%'.format(error_magn*100))
print('Velocity error in u:  {0:.3f}%'.format(error_u*100))
print('Velocity error in v:  {0:.3f}%'.format(error_v*100))

# we now check the out-of sample results.
# Get out of sample position
U_out=U_p[indices_test];  V_out=V_p[indices_test]
X_out=X_p[indices_test];  Y_out=Y_p[indices_test]

# Out of sample predictions
U_calc_O,V_calc_O=SP_vel.Get_Sol([X_out,Y_out])

# Magnitude of the RBF solution
U_magn_calc_O = np.sqrt(U_calc_O**2 + V_calc_O**2)
# Compute the magnitude of the analytical solution
U_magn_O = np.sqrt(U_out**2 + V_out**2)
# Compute the error in the magnitude
error_magn_O = np.linalg.norm(U_magn_calc_O - U_magn_O)/np.linalg.norm(U_magn_O)
# Error in u
error_u_O = np.linalg.norm(U_calc_O - U_out) / np.linalg.norm(U_out)
# Error in v
error_v_O = np.linalg.norm(V_calc_O - V_out) / np.linalg.norm(V_out)


print('------- Out of sample error results ---------')
print('Total velocity error: {0:.3f}%'.format(error_magn_O*100))
print('Velocity error in u:  {0:.3f}%'.format(error_u_O*100))
print('Velocity error in v:  {0:.3f}%'.format(error_v_O*100))

# fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(15,6), dpi=100, sharex=True, sharey=True)
# axes[0,0].scatter(X, Y, c=U_calc, s=10)
# axes[1,0].scatter(X, Y, c=U, s=10)
# axes[2,0].scatter(X, Y, c=np.abs(U_calc-U), s=10) 

# axes[0,1].scatter(X, Y, c=V_calc, s=10)
# axes[1,1].scatter(X, Y, c=V, s=10)
# axes[2,1].scatter(X, Y, c=np.abs(V_calc-V), s=10)  

# axes[0,2].scatter(X, Y, c=U_magn_calc, s=10)
# axes[1,2].scatter(X, Y, c=U_magn, s=10)
# axes[2,2].scatter(X, Y, c=np.abs(U_magn_calc-U_magn), s=10) 


# axes[0,0].set_ylabel('RBF Regression') 
# axes[1,0].set_ylabel('Ground truth')  
# axes[2,0].set_ylabel('Absolute difference')  

# axes[0,0].set_title('$u$') 
# axes[0,1].set_title('$v$')  
# axes[0,2].set_title('$|\mathbf{u}-\mathbf{u}_c|$')      
# for ax in axes.flatten():
#     ax.set_aspect(1)      
# fig.tight_layout()      

# Number of constraints on each boundary
n_c = 100

################################# Define the Location for the Neumann conditions (5 patches) #################
# We start with the Neumann conditions
# Left boundary
X_Pres_N1 = np.zeros(n_c)
Y_Pres_N1 = np.linspace(0, H, n_c)
# Bottom boundary
X_Pres_N2 = np.linspace(0.0,L,n_c)
Y_Pres_N2 = np.zeros(n_c)
# Top boundary
X_Pres_N4 = np.linspace(0, L, n_c)
Y_Pres_N4 = np.ones(n_c)*H
# Cylinder boundary
alpha_P = np.linspace(0, 2*np.pi, n_c, endpoint = False) 
X_Pres_N5 = 0.2 + R*np.cos(alpha_P)
Y_Pres_N5 = 0.2 + R*np.sin(alpha_P)
# Assemble the the entire array of Neumann points
X_Pres_N=np.hstack((X_Pres_N1, X_Pres_N2, X_Pres_N4, X_Pres_N5))
Y_Pres_N=np.hstack((Y_Pres_N1, Y_Pres_N2, Y_Pres_N4, Y_Pres_N5))

############################### Compute the normal vectors for all patches requireing N conditions #################

# We assemble the normals in the same way
# Left boundary
n_x_1 = np.ones(X_Pres_N1.shape)*(-1)
n_y_1 = np.ones(X_Pres_N1.shape)*0
# Bottom boundary
n_x_2 = np.ones(X_Pres_N2.shape)*0
n_y_2 = np.ones(X_Pres_N2.shape)*(-1)
# Top boundary
n_x_4 = np.ones(X_Pres_N4.shape)*0
n_y_4 = np.ones(X_Pres_N4.shape)*1
# Cylinder boundary
n_x_5 = np.ones(X_Pres_N5.shape)*(-np.cos(alpha_P))
n_y_5 = np.ones(X_Pres_N5.shape)*(-np.sin(alpha_P))
# Assemble to obtain the entire array of Neumann normals
n_x = np.hstack((n_x_1, n_x_2, n_x_4, n_x_5))
n_y = np.hstack((n_y_1, n_y_2, n_y_4, n_y_5))  


############### Clean for possible repeaded points (usually along corners ) ############################################
_, valid_idcs = np.unique(np.column_stack((X_Pres_N, Y_Pres_N)),
                          return_index = True, axis = 0)
X_Pres_N = X_Pres_N[valid_idcs]
Y_Pres_N = Y_Pres_N[valid_idcs]
n_x = n_x[valid_idcs]
n_y = n_y[valid_idcs]

######################### Define location (and value) for the Dirichlet (D) condition at the outlet ###################

X_Pres_D3 = np.ones(n_c-2)*L
Y_Pres_D3 = np.linspace(0, H, n_c)[1:-1]

#################### Define the location of the pressure probe from which we will take another D condition #################

x_loc=0.4; y_loc=0.2    ### PLAY WITH THIS !! # this is the approximate location of the pressure prob.

# look for the close point
x_err=(X_p-x_loc)**2+(Y_p-y_loc)**2
index_probe=np.argmin(x_err)

# Define the Dirichlet condition associated to the probe.
X_S=X_p[index_probe]
Y_S=Y_p[index_probe]
P_S=P_p[index_probe]

# thus the full set of D conditions is :
X_Pres_D = np.append(X_Pres_D3,X_S)
Y_Pres_D = np.append(Y_Pres_D3,Y_S)
P_Pres_D = np.hstack([np.zeros(X_Pres_D3.shape),P_S])

# First we compute the required quantities from the velocity field 
# (neither of the following 2 steps runs if SP_Vel has not been solved)
# 1. Evaluate the source term on the RHS of the Poisson equation
source_term = SP_vel.Evaluate_Source_Term(grid=[X,Y], rho=rho)
# 2. Evaluate the c_N for the N conditions (see Presentation 1)
P_Neu = SP_vel.Get_Pressure_Neumann(grid = [X_Pres_N, Y_Pres_N], 
                                    normals = [n_x, n_y],
                                    rho = rho, mu = mu)

# We can now proceed with (1) spicy initialization (2) clustering (3) constraint assingment, (4) System Assembly:

# We assemble our Neumann and Dirichlet B.C.
NEU_P = [X_Pres_N, Y_Pres_N, n_x, n_y, P_Neu]
DIR_P = [X_Pres_D, Y_Pres_D, P_Pres_D]

SP_pres = spicy([source_term], [X,Y], basis='gauss')
SP_pres.clustering([3,10,50,200,1000], Areas=[poly1,poly2,[],[],[]], r_mM=[0.015,0.5], eps_l=0.87)

# And, we set them
SP_pres.scalar_constraints(DIR=DIR_P, 
                           NEU=NEU_P, 
                           extra_RBF=True)

# SP_pres.plot_RBFs(l=4) # Plot the clustering a level 

SP_pres.Assembly_Poisson() # Assembly the system

SP_pres.Solve(K_cond=1e9) # Solve the system

# Check the results on the 'training data'
P_calc = SP_pres.Get_Sol(grid=[X,Y])
# print the pressure error
error_p = np.linalg.norm(P_calc-P)/np.linalg.norm(P)
print('------- in sample pressure error results ---------')
print('Total pressure error: {0:.3f}%'.format(error_p*100))

# Check the results on the 'testing data'
P_calc_test = SP_pres.Get_Sol(grid=[X_p[indices_test],Y_p[indices_test]])
P_test=P_p[indices_test]
error_p = np.linalg.norm(P_calc_test-P_test)/np.linalg.norm(P_test)
print('------- out of sample pressure error results ---------')
print('Total pressure error: {0:.3f}%'.format(error_p*100))



# fig, axes = plt.subplots(nrows=3, dpi=100)
# axes[0].set_title('Original Pressure')
# sc=axes[0].scatter(X, Y, c=P)
# plt.colorbar(sc,ax=axes[0])
# axes[0].plot(X_S,Y_S,'ro')
# axes[1].set_title('Reconstructed Pressure')
# sc2=axes[1].scatter(X, Y, c=P_calc)
# plt.colorbar(sc2,ax=axes[1])
# axes[2].set_title('Difference')
# sc3=axes[2].scatter(X, Y, c=P_calc-P)
# plt.colorbar(sc3,ax=axes[2])


# for ax in axes.flatten():
#     ax.set_aspect(1)
# fig.tight_layout()

