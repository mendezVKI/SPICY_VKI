# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 12:08:40 2023

@author: mendez
"""

import numpy as np
import matplotlib.pyplot as plt 
import scipy.special as sc 
from spicy_class_m import spicy 

# This is for plot customization
fontsize = 16
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['axes.labelsize'] = fontsize
plt.rcParams['legend.fontsize'] = fontsize
plt.rcParams['font.size'] = fontsize



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


# Step 1: Initialize Spicy Class
SP_vel = spicy([U_noise,V_noise], [X,Y], basis='c4')

# Step 2: Perform the clustering
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
DIV = [X_Div, Y_Div]

# We set the constraints in these points and also place additional RBFs in each of these points|
SP_vel.vector_constraints(DIV=DIV, extra_RBF=True)







