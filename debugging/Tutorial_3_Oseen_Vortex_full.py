# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:54:28 2023

@author: Ratz
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.special as sc 
from spicy_class_2103 import spicy 

# This is for plot customization
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=14)
plt.rc('ytick',labelsize=14)

# Fix the random seed to ensure reproducibility
np.random.seed(42)

# Number of particles
n_p = 5000

# Define the domain boundaries and flow properties
x1_hat, x2_hat = -0.5, 0.5 # m, m
y1_hat, y2_hat = -0.5, 0.5 # m, m
rho = 1 # kg/m^3
mu = 0 # Pa s

# Generate the random points
X = np.random.random(n_p)*(x2_hat - x1_hat) + x1_hat
Y = np.random.random(n_p)*(y2_hat - y1_hat) + y1_hat

# Compute the radius and angle in the 2D domain
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# Hyperparameters of the vortex
Gamma = 10
r_c = 0.1
gamma = 1.256431
c_theta = r_c**2/gamma

# Compute the velocity field
u_theta = Gamma / (2*np.pi*r) * (1 - np.exp(-r**2 / (r_c**2 / gamma)))
U = np.sin(theta) * u_theta
V = -np.cos(theta) * u_theta 

# Add 10% noise to it
q = 0.1
U_noise = U * (1 + q*np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q*np.random.uniform(-1, 1, size = V.shape))

SP_vel = spicy([U_noise,V_noise], [X,Y], basis='gauss')

SP_vel.clustering([6,50], r_mM=[0.05,0.7], eps_l=0.88)

# Number of points for the vertical and horizontal boundary
n_c_V = n_c_H = 50

# Left boundary
X_Div1 = np.ones(n_c_V)*(x1_hat)
Y_Div1 = np.linspace(y1_hat, y2_hat, n_c_V)
# Bottom boundary
X_Div2 = np.linspace(x1_hat, x2_hat, n_c_H)
Y_Div2 = np.ones(n_c_H)*y1_hat
# Right boundary
X_Div3 = np.ones(n_c_V)*x2_hat
Y_Div3 = np.linspace(y1_hat, y2_hat, n_c_V)
# Top boundary
X_Div4 = np.linspace(x1_hat, x2_hat, n_c_H)
Y_Div4 = np.ones(n_c_H)*y2_hat

# Assemble to obtain the entire array of boundary conditions
X_Div=np.hstack((X_Div1,X_Div2,X_Div3,X_Div4))
Y_Div=np.hstack((Y_Div1,Y_Div2,Y_Div3,Y_Div4))

# we remove the duplicates
_, valid_idcs = np.unique(np.column_stack((X_Div, Y_Div)), return_index = True, axis = 0)
X_Div = X_Div[valid_idcs]
Y_Div = Y_Div[valid_idcs]

# We set the constraints in these points and also place additional RBFs in each of these points|
SP_vel.vector_constraints(DIV=[X_Div, Y_Div], extra_RBF=True)

SP_vel.Assembly_Regression(n_hb = 0, alpha_div = 1)

SP_vel.Solve(K_cond=1e8)

solution_P=SP_vel.Get_Sol(grid = [X,Y])
U_P = solution_P[:n_p]
V_P = solution_P[n_p:]

# Magnitude of the RBF solution
U_magn_fit = np.sqrt(U_P**2 + V_P**2)
# Compute the magnitude of the analytical solution
U_magn_corr = np.sqrt(U**2 + V**2)
# Compute the error in the magnitude
error_magn = np.linalg.norm(U_magn_fit - U_magn_corr) / np.linalg.norm(U_magn_corr)

print('Total velocity error: {0:.3f}%'.format(error_magn*100))

fig, axes = plt.subplots(figsize = (15, 5), dpi = 100, ncols = 3)
axes[0].scatter(X, Y, c = U_magn_fit)
axes[0].set_title('Computed velocity field')
axes[1].scatter(X, Y, c = U_magn_corr)
axes[1].set_title('Analytical velocity field')
axes[2].scatter(X, Y, c = np.abs(U_magn_corr - U_magn_fit))
axes[2].set_title('Absolute difference')             
for ax in axes.flatten():
    ax.set_aspect(1)

#%% Here, the pressure computation starts

# Get the source term
source_term = SP_vel.Evaluate_Source_Term(grid = [X, Y], rho = rho)

# Get the one dirichlet condition from the analytical solution
X_pres = -0.5; Y_pres = -0.5
radius_pres = np.sqrt(X_pres**2+Y_pres**2)
u_theta_pres = Gamma/(2*np.pi*radius_pres)*(1-np.exp(-radius_pres**2/c_theta))
pres_dir = -np.array([0.5*rho*(u_theta_pres)**2-\
                rho*Gamma**2/(4*np.pi**2*c_theta)*(sc.exp1(radius_pres**2/c_theta)-sc.exp1(2*radius_pres**2/c_theta))])

# Number of points for the vertical and horizontal boundary
n_c_V = n_c_H = 50
# Left boundary
X_Pres_N1=np.ones(n_c_V)[1:]*(x1_hat) # we remove the point where our Dirichlet condition is
Y_Pres_N1=np.linspace(y1_hat,y2_hat,n_c_V)[1:] # we remove the point where our Dirichlet condition is
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
n_x_1 = np.ones(n_c_V)[1:]*(-1)
n_y_1 = np.ones(n_c_V)[1:]*0
# Bottom boundary
n_x_2 = np.ones(n_c_H)[1:]*0
n_y_2 = np.ones(n_c_H)[1:]*(-1)
# Right boundary
n_x_3 = np.ones(n_c_V)*1
n_y_3 = np.ones(n_c_V)*0
# Top boundary
n_x_4 = np.ones(n_c_H)*0
n_y_4 = np.ones(n_c_H)*(1)

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

# Evaluate the pressure in these points
P_Neu = SP_vel.Get_Pressure_Neumann(grid = [X_Pres_N, Y_Pres_N], normals = [n_x, n_y],
                                    rho = rho, mu = mu)

SP_pres = spicy([source_term], [X, Y], basis = 'gauss')

SP_pres.clustering([6,50], r_mM=[0.05, 0.7], eps_l=0.88)

# We assemble our Neumann and Dirichlet B.C.
NEU_P = [X_Pres_N, Y_Pres_N, n_x, n_y, P_Neu]
DIR_P = [np.array([X_pres]), np.array([Y_pres]), pres_dir]

# And, we set them
SP_pres.scalar_constraints(DIR = DIR_P, NEU = NEU_P, extra_RBF = True)

SP_pres.Assembly_Poisson(n_hb = 0)

SP_pres.Solve(K_cond = 1e8)

P_calc = SP_pres.Get_Sol(grid = [X, Y])

# Compute the analytical pressure field
u_theta = Gamma/(2*np.pi*r)*(1-np.exp(-r**2/c_theta))
P_corr = -0.5*rho*u_theta**2-rho*Gamma**2/(4*np.pi**2*c_theta)*(sc.exp1(r**2/c_theta)-sc.exp1(2*r**2/c_theta))

# print the pressure error
error_p = np.linalg.norm(P_calc-P_corr)/np.linalg.norm(P_corr)
print('Total pressure error: {0:.3f}%'.format(error_p*100))

fig, axes = plt.subplots(figsize = (15, 5), dpi = 100, ncols = 3)
axes[0].scatter(X, Y, c = P_calc)
axes[0].set_title('Computed pressure field')
axes[1].scatter(X, Y, c = P_corr)
axes[1].set_title('Analytical pressure field')
axes[2].scatter(X, Y, c = np.abs(P_calc - P_corr))
axes[2].set_title('Absolute difference')             
for ax in axes.flatten():
    ax.set_aspect(1)
