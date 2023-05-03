# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 15:27:43 2023

@author: Ratz, Mendez
"""


import numpy as np
import matplotlib.pyplot as plt 
import scipy.io
from spicy_class_m import spicy

# This is for plot customization
fontsize = 12
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['font.size'] = fontsize

#%% 1. Create the dataset.

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

# From the remaining points we can choose to sample a random amount of points
# if we want to go for a smaller test case. In this
# tutorial, we take a maximum of 18755 points. We will use 70% of these and 
# keep 20% to evaluate the RBF prediction performances in out of sample conditions


# A testing data, we take the ones that are not in this list.
from sklearn.model_selection import train_test_split
indices_train, indices_test=train_test_split(np.arange(0,len(X_p),1),test_size=0.2)

n_p = len(indices_train)

# Select the data points
X = X_p[indices_train]; Y = Y_p[indices_train]; P = P_p[indices_train]
U = U_p[indices_train]; V = V_p[indices_train]

# Add 0.3 noise to the velocity field
q = 0.05
U_noise = U * (1 + q * np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q * np.random.uniform(-1, 1, size = V.shape))


# Get indices of the points
indices=np.linspace(0,len(X)-1,len(X))


#%% 2. Define all the BC's for constraints

# Number of constraints on each boundary
n_c = 50

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

# We set the constraints in these points and also place additional RBFs in each of these points

#%% New feature: polygons!!

# We perform the clustering using two polygons
from shapely import geometry

p1 = geometry.Point(0.1,0.1); p2 = geometry.Point(0.3,0.1)
p3 = geometry.Point(0.3,0.3); p4 = geometry.Point(0.1,0.3)

pointList = [p1, p2, p3, p4]
poly1 = geometry.Polygon([i for i in pointList])

plt.scatter(X,Y,c=P)
plt.plot(*poly1.exterior.xy,'r') # in case you want to see them for info

# Define also a second box
p1 = geometry.Point(0.0,0); p2 = geometry.Point(0.5,0)
p3 = geometry.Point(0.5,0.45); p4 = geometry.Point(0.0,0.45)

pointList = [p1, p2, p3, p4]
poly2 = geometry.Polygon([i for i in pointList])

plt.plot(*poly2.exterior.xy,'r') # in case you want to see them for info

List_Poly=[poly1,poly2,[],[]] # the third scale acts on the full domain


# Prepare the SPICY object
SP_vel = spicy([U_noise,V_noise], [X,Y], basis='gauss')
# Clustering 
SP_vel.clustering([3,6,50,200], Areas=List_Poly, r_mM=[0.015,0.3], eps_l=0.87)
# Introduce the Constraints
SP_vel.vector_constraints(DIR=DIR, DIV=DIV, extra_RBF=True)
# Plot the RBF cluster
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








#%% Plot the results

fig, axes = plt.subplots(ncols=3, nrows=3, figsize=(15,6), dpi=100, sharex=True, sharey=True)
axes[0,0].scatter(X, Y, c=U_calc, s=10)
axes[1,0].scatter(X, Y, c=U, s=10)
axes[2,0].scatter(X, Y, c=np.abs(U_calc-U), s=10) 

axes[0,1].scatter(X, Y, c=V_calc, s=10)
axes[1,1].scatter(X, Y, c=V, s=10)
axes[2,1].scatter(X, Y, c=np.abs(V_calc-V), s=10)  

axes[0,2].scatter(X, Y, c=U_magn_calc, s=10)
axes[1,2].scatter(X, Y, c=U_magn, s=10)
axes[2,2].scatter(X, Y, c=np.abs(U_magn_calc-U_magn), s=10) 


axes[0,0].set_ylabel('RBF Regression') 
axes[1,0].set_ylabel('Ground truth')  
axes[2,0].set_ylabel('Absolute difference')  

axes[0,0].set_title('$u$') 
axes[0,1].set_title('$v$')  
axes[0,2].set_title('$||\mathbf{u}||_2^2$')      
for ax in axes.flatten():
    ax.set_aspect(1)      
fig.tight_layout()      


#%% Pressure Computation
source_term = SP_vel.Evaluate_Source_Term(grid=[X,Y], rho=rho)

# Number of constraints on each boundary
n_c = 100

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
# Top boundary
n_x_4 = np.ones(X_Pres_N4.shape)*0
n_y_4 = np.ones(X_Pres_N4.shape)*1
# Cylinder boundary
n_x_5 = np.ones(X_Pres_N5.shape)*(-np.cos(alpha_P))
n_y_5 = np.ones(X_Pres_N5.shape)*(-np.sin(alpha_P))
# Assemble to obtain the entire array of Neumann normals
n_x = np.hstack((n_x_1, n_x_2, n_x_4, n_x_5))
n_y = np.hstack((n_y_1, n_y_2, n_y_4, n_y_5))  

# Remove the duplicates for the normals
_, valid_idcs = np.unique(np.column_stack((X_Pres_N, Y_Pres_N)),
                          return_index = True, axis = 0)
X_Pres_N = X_Pres_N[valid_idcs]
Y_Pres_N = Y_Pres_N[valid_idcs]
n_x = n_x[valid_idcs]
n_y = n_y[valid_idcs]

# The last thing are the Dirichlet conditions at the outlet
# Right boundary
X_Pres_D3 = np.ones(n_c-2)*L
Y_Pres_D3 = np.linspace(0, H, n_c)[1:-1]

# or we add just one pressure prob at the inlet:
#X_Pres_D3=
#Y_Pres_D3=    
    
    

# We also assume that we have 1 pressure tap somewhere
X_S=X[6991]
Y_S=Y[6991]
P_S=P[6991]


#%% Proceed with the integration 

# Evaluate the Neumann conditions in these points
P_Neu = SP_vel.Get_Pressure_Neumann(grid = [X_Pres_N, Y_Pres_N], 
                                    normals = [n_x, n_y],
                                    rho = rho, mu = mu)

# The Dirichlet conditions do not have any overlap with the Neumann conditions, so we can just take them as they are
X_Pres_D = np.append(X_Pres_D3,X_S)
Y_Pres_D = np.append(Y_Pres_D3,Y_S)
P_Pres_D = np.hstack([np.zeros(X_Pres_D3.shape),P_S])

# X_Pres_D = X_Pres_D3
# Y_Pres_D = Y_Pres_D3
# P_Pres_D = np.zeros(X_Pres_D.shape)

List_Poly=[poly1,poly2,[],[],[]]

SP_pres = spicy([source_term], [X,Y], basis='gauss')
SP_pres.clustering([3,10,50,200,1000], Areas=List_Poly, r_mM=[0.015,0.5], eps_l=0.87)

# We assemble our Neumann and Dirichlet B.C.
NEU_P = [X_Pres_N, Y_Pres_N, n_x, n_y, P_Neu]
DIR_P = [X_Pres_D, Y_Pres_D, P_Pres_D]

# And, we set them
SP_pres.scalar_constraints(DIR=DIR_P, 
                           NEU=NEU_P, 
                           extra_RBF=True)

SP_pres.plot_RBFs(l=4)


SP_pres.Assembly_Poisson(n_hb = 0)
SP_pres.Solve(K_cond=1e7)
P_calc = SP_pres.Get_Sol(grid=[X,Y])
# print the pressure error
error_p = np.linalg.norm(P_calc-P)/np.linalg.norm(P)
print('Total pressure error: {0:.3f}%'.format(error_p*100))


fig, axes = plt.subplots(nrows=3, dpi=100)
axes[0].set_title('Original Pressure')
sc=axes[0].scatter(X, Y, c=P)
plt.colorbar(sc)
axes[1].set_title('Reconstructed Pressure')
sc2=axes[1].scatter(X, Y, c=P_calc)
plt.colorbar(sc2)
axes[2].set_title('Difference')
sc3=axes[2].scatter(X, Y, c=P_calc-P)
plt.colorbar(sc3)


for ax in axes.flatten():
    ax.set_aspect(1)
fig.tight_layout()





