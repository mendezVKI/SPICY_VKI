# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:54:28 2023

@author: Ratz
"""

import numpy as np
import matplotlib.pyplot as plt 
from spicy_class_2103 import spicy 

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

SP_vel = spicy([U_noise], [X,Y], basis='gauss')

SP_vel.clustering([6,50], r_mM=[0.05,0.7], eps_l=0.88)

SP_vel.scalar_constraints()

SP_vel.plot_RBFs()

SP_vel.Assembly_Regression(n_hb = 0)

SP_vel.Solve(K_cond=1e8)

solution_velocity = SP_vel.Get_Sol(grid=[X,Y])
# Extract individual velocity components
U_calc = solution_velocity

error_u = np.linalg.norm(U_calc - U) / np.linalg.norm(U)
print('Velocity error in  u: {0:.3f}%'.format(error_u*100))

SP_vel.u = V_noise

# Assemble the system
SP_vel.Assembly_Regression(n_hb = 0)
# Solve it
SP_vel.Solve(K_cond=1e8)
# And obtain the solution of V
solution_velocity = SP_vel.Get_Sol(grid=[X,Y])
# Extract individual velocity components
V_calc = solution_velocity
# Compute the error and print it
error_v = np.linalg.norm(V_calc - V) / np.linalg.norm(V)
print('Velocity erro in v: {0:.3f}%'.format(error_v*100))

# Magnitude of the RBF solution
U_magn_calc = np.sqrt(U_calc**2 + V_calc**2)
# Compute the magnitude of the analytical solution
U_magn = np.sqrt(U**2 + V**2)
# Compute the error in the magnitude
error_magn = np.linalg.norm(U_magn_calc - U_magn) / np.linalg.norm(U_magn)

print('Total velocity error: {0:.3f}%'.format(error_magn*100))

fig, ax = plt.subplots(ncols = 3, nrows = 3, figsize = (12, 12), dpi = 100,
                       sharex = True, sharey = True)
ax[0,0].scatter(X, Y, c=U_calc, s=10)
ax[1,0].scatter(X, Y, c=U, s=10)
ax[2,0].scatter(X, Y, c=np.abs(U_calc-U), s=10) 

ax[0,1].scatter(X, Y, c=V_calc, s=10)
ax[1,1].scatter(X, Y, c=V, s=10)
ax[2,1].scatter(X, Y, c=np.abs(V_calc-V), s=10)  


ax[0,2].scatter(X, Y, c=U_magn_calc, s=10)
ax[1,2].scatter(X, Y, c=U_magn, s=10)
ax[2,2].scatter(X, Y, c=np.abs(U_magn_calc-U_magn), s=10) 


ax[0,0].set_ylabel('Computed velocity field') 
ax[1,0].set_ylabel('Ground truth velocity field')  
ax[2,0].set_ylabel('Absolute difference')  

ax[0,0].set_title('$u$') 
ax[0,1].set_title('$v$')  
ax[0,2].set_title('$||\mathbf{u}||_2^2$')        
fig.tight_layout()