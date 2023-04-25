# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:13:04 2023

@author: ManuelRatz
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
from spicy_class_2103 import spicy

# This is for plot customization
fontsize = 20
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize

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
X = data['x'].reshape(-1)
Y = data['y'].reshape(-1) 
# Extract the velocities
U = data['vx'].reshape(-1)
V = data['vy'].reshape(-1)
P = data['p'].reshape(-1)

# Here, we remove the points at the inlet and at the wall, as they are given by the constraints
inlet_and_wall_remover = np.invert(np.logical_or(np.logical_and(U==0, V==0), X==0))
# Remove the points
X = X[inlet_and_wall_remover]
Y = Y[inlet_and_wall_remover]
P = P[inlet_and_wall_remover]
U = U[inlet_and_wall_remover]
V = V[inlet_and_wall_remover]

# From the remaining points we can choose to sample a random amount if we want to go for a smaller test case. In this
# tutorial, we take the maximum number of points which is 18755
n_p = 18755
random_points_indices = np.random.randint(low=0, high=len(X), size=n_p)
# Select the data points
X = X[random_points_indices]
Y = Y[random_points_indices]
P = P[random_points_indices]
U = U[random_points_indices]
V = V[random_points_indices]

# Add 10% noise to the velocity field
q = 0.1
U_noise = U * (1 + q * np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q * np.random.uniform(-1, 1, size = V.shape))

# define the basis
basis = 'gauss'

# Step 1: Initialize the spicy class for the velocity regression
SP_vel = spicy([U_noise, V_noise], [X,Y], basis=basis)

# Step 2: Perform the clustering
SP_vel.clustering([6,50,1800], r_mM=[0.015,0.5], eps_l=0.83)


# Number of constraints on each boundary
n_c = 150 

# Left boundary
X_Div1 = np.zeros(n_c)
Y_Div1 = np.linspace(0, H, n_c)
U_Dir1 = 4*(H-Y_Div1)*Y_Div1/H**2
V_Dir1 = np.zeros(X_Div1.shape)
# Bottom boundary
X_Div2 = np.linspace(0, L, n_c)
Y_Div2 = np.zeros(n_c)
U_Dir2 = np.zeros(X_Div2.shape)
V_Dir2 = np.zeros(X_Div2.shape)
# Right boundary
X_Div3 = np.ones(n_c-2)*L
Y_Div3 = np.linspace(0, H, n_c)[1:-1]
# Top boundary
X_Div4 = np.linspace(0, L, n_c)
Y_Div4 = np.ones(n_c)*H
U_Dir4 = np.zeros(X_Div4.shape)
V_Dir4 = np.zeros(X_Div4.shape)
# Cylinder boundary
alphaT = np.linspace(0, 2*np.pi, n_c, endpoint = False)
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

# TODO This is for debugging of duplicate removal only
n_c_V = n_c_H = 50
x1_hat = 0; x2_hat = 1
y1_hat = 0; y2_hat = 4.1
# Left boundary
X_Pres_N1=X_Div1 # we remove the point where our Dirichlet condition is
Y_Pres_N1=Y_Div1# we remove the point where our Dirichlet condition is
# Bottom boundary
X_Pres_N2=np.linspace(x1_hat,x2_hat,n_c_H)[1:]
Y_Pres_N2=np.ones(n_c_H)[1:]*y1_hat
# Right boundary
X_Pres_N3=np.ones(n_c_V)*x2_hat
Y_Pres_N3=np.linspace(y1_hat,y2_hat,n_c_V)
# Top boundary
X_Pres_N4=np.linspace(x1_hat,x2_hat,n_c_H)
Y_Pres_N4=np.ones(n_c_H)*y2_hat

# Assemble the individual arrays
X_Pres_N=np.hstack((X_Pres_N1,X_Pres_N2,X_Pres_N3,X_Pres_N4))
Y_Pres_N=np.hstack((Y_Pres_N1,Y_Pres_N2,Y_Pres_N3,Y_Pres_N4))

# we assemble the normals in the same way
# Left boundary
n_x_1 = np.ones(X_Pres_N1.shape)*(-1)
n_y_1 = np.ones(X_Pres_N1.shape)*0
# Bottom boundary
n_x_2 = np.ones(X_Pres_N2.shape)*0
n_y_2 = np.ones(X_Pres_N2.shape)*(-1)
# Right boundary
n_x_3 = np.ones(X_Pres_N3.shape)*1
n_y_3 = np.ones(X_Pres_N3.shape)*0
# Top boundary
n_x_4 = np.ones(X_Pres_N4.shape)*0
n_y_4 = np.ones(X_Pres_N4.shape)*(1)

# Assemble to obtain the entire array of normals
n_x = np.hstack((n_x_1, n_x_2, n_x_3, n_x_4))
n_y = np.hstack((n_y_1, n_y_2, n_y_3, n_y_4)) 

# Remove the duplicates again
_, valid_idcs = np.unique(np.column_stack((X_Pres_N, Y_Pres_N)),
                          return_index = True, axis = 0)
X_Pres_N = X_Pres_N[valid_idcs]
Y_Pres_N = Y_Pres_N[valid_idcs]
n_x = n_x[valid_idcs]
n_y = n_y[valid_idcs]
U_Neu = np.ones(X_Pres_N.shape)*0.5
V_Neu = np.ones(Y_Pres_N.shape)*0.5
NEU = [X_Pres_N, Y_Pres_N, n_x, n_y, U_Neu, V_Neu]

# We set the constraints in these points and also place additional RBFs in each of these points
SP_vel.vector_constraints(DIR = DIR, DIV = DIV, extra_RBF = True)

SP_vel.plot_RBFs()

# Step 4. Assemble the linear system
SP_vel.Assembly_Regression(n_hb = 0, alpha_div = 1)

# Step 5. Solve the linear system
SP_vel.Solve(K_cond=1e12)
# SP_vel.Solve_Pietro(rcond=1e-12)

# Step 6. Get the solution on the scattered data points again
solution_velocity = SP_vel.Get_Sol(grid=[X,Y])
# Extract individual velocity components
U_calc = solution_velocity[:n_p]
V_calc = solution_velocity[n_p:]

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

print('Total velocity error: {0:.3f}%'.format(error_magn*100))
print('Velocity error in u:  {0:.3f}%'.format(error_u*100))
print('Velocity error in v:  {0:.3f}%'.format(error_v*100))
#%%
source_term = SP_vel.Evaluate_Source_Term(grid=[X,Y], rho=rho)

# Number of constraints on each boundary
n_c = 150 

# Right boundary
X_Pres_D3 = np.ones(n_c-2)*L
Y_Pres_D3 = np.linspace(0, H, n_c)[1:-1]

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

# We assemble the normals in the same way
# Left boundary
n_x_1 = np.ones(X_Pres_N1.shape)*(-1)
n_y_1 = np.ones(X_Pres_N1.shape)*0
# Bottom boundary
n_x_2 = np.ones(X_Pres_N2.shape)*0
n_y_2 = np.ones(X_Pres_N2.shape)*(-1)
# Right boundary
n_x_4 = np.ones(X_Pres_N4.shape)*0
n_y_4 = np.ones(X_Pres_N4.shape)*1
# Top boundary
n_x_5 = np.ones(X_Pres_N5.shape)*(-np.cos(alpha_P))
n_y_5 = np.ones(X_Pres_N5.shape)*(-np.sin(alpha_P))
# Assemble to obtain the entire array of Neumann normals
n_x = np.hstack((n_x_1, n_x_2, n_x_4, n_x_5))
n_y = np.hstack((n_y_1, n_y_2, n_y_4, n_y_5))  

# Remove the duplicates again
_, valid_idcs = np.unique(np.column_stack((X_Pres_N, Y_Pres_N)),
                          return_index = True, axis = 0)
X_Pres_N = X_Pres_N[valid_idcs]
Y_Pres_N = Y_Pres_N[valid_idcs]
n_x = n_x[valid_idcs]
n_y = n_y[valid_idcs]

# Evaluate the pressure in these points
P_Neu = SP_vel.Get_Pressure_Neumann(grid = [X_Pres_N, Y_Pres_N], normals = [n_x, n_y],
                                    rho = rho, mu = mu)

# The Dirichlet conditions do not have any overlap with the Neumann conditions, so we can just take them as they are
X_Pres_D = X_Pres_D3
Y_Pres_D = Y_Pres_D3
P_Pres_D = np.zeros(X_Pres_D.shape)

NEU_P = [X_Pres_N, Y_Pres_N, n_x, n_y, P_Neu]
DIR_P = [X_Pres_D, Y_Pres_D, P_Pres_D]

# 1.: Define the class
SP_pres = spicy([source_term], [X, Y], basis = basis)

# 2.: Perform the clustering
# We take the same one as for the velocity regression, I did not yet test
# what the benefits of varying the clustering might be
SP_pres.clustering([6,50,1800], r_mM=[0.015, 0.5], eps_l=0.83)

# 3.: Set boundary conditions
SP_pres.scalar_constraints(DIR = DIR_P, NEU = NEU_P, extra_RBF = True)
# SP_pres.plot_RBFs()

# 4. Assemble the linear system: 
SP_pres.Assembly_Poisson(n_hb = 0)

# 5. Solve the system
SP_pres.Solve(K_cond = 1e12)

# We compute the pressure on the scattered points
P_calc = SP_pres.Get_Sol(grid = [X, Y])

# print the pressure error
error_p = np.linalg.norm(P_calc-P)/np.linalg.norm(P)
print('Total pressure error: {0:.3f}%'.format(error_p*100))

fig, axes = plt.subplots(figsize=(15,3), dpi=100, ncols=3, sharey=True)
axes[0].scatter(X, Y, c=P_calc, s=10)
axes[0].set_title('Computed pressure field')
axes[1].scatter(X, Y, c=P, s=10)
axes[1].set_title('Analytical pressure field')
axes[2].scatter(X, Y, c=np.abs(P_calc-P), s=10)
axes[2].set_title('Absolute difference')           
for ax in axes.flatten():
    ax.set_aspect(1)
fig.tight_layout()
