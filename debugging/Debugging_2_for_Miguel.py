# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 17:54:28 2023

@author: Ratz
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd # this is needed to remove duplicates
import scipy.special as sc # this is needed for the evaluation of the analytical pressure
# this is the file that I am editing at the moment, when you approve the changes,
# we can update the class
from spicy_class_2702 import spicy 

# Set the random seeds
rnd = np.random.default_rng(seed=39)
rndu = np.random.default_rng(seed=47)
rndv = np.random.default_rng(seed=42)

# number of particles
n_p = 2000
# This would be the sampling in the real domain
np.random.seed(42)

# Define the domain from -0.5 to 0.5 (as this is also what is done in the paper)
x1,x2 = -0.5, 0.5
y1,y2 = -0.5, 0.5
x1_hat,x2_hat = -0.5, 0.5
y1_hat,y2_hat = -0.5, 0.5

# generate X and Y points
X = rnd.random(n_p)*(x2 - x1) + x1
Y = rnd.random(n_p)*(y2 - y1) + y1
# compute radius and angle for the Oseen vortex
r = np.sqrt(X**2 + Y**2)
theta = np.arctan2(Y, X)

# 'Hyperparameters' of the vortex, taken from the paper
Gamma = 10
rc = 0.1
gamma = 1.256431
rho=1
mu=0
cTH=rc**2/gamma

# analytical velocity field. Important to note is that this is not rescaled.
# This happens internally during the regression. Otherwise, I was having problems
# when computing the pressure field as the units were messed up. There is probably
# a better way to do this, so I am very open to suggestions
u_theta = Gamma / (2*np.pi*r) * (1 - np.exp(-r**2 / (rc**2 / gamma)))
U = np.sin(theta) * u_theta
V = -np.cos(theta) * u_theta 

# add 10 % noise
q = 0.1
U_noise = U * (1 + q * np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q * np.random.uniform(-1, 1, size = V.shape))

# we define the basis here (we do it twice, just to be sure that both the
# velocity regression and pressure integration use the same)
basis = 'c4'

# Step 1. Create the class of the velocity
SP_vel = spicy([U_noise, V_noise],[X,Y],basis=basis)

# Here, I was testing different clustering options, depending on the number of 
# particles and the domain size
# TODO I need to check this again, but I got higher errors when the domain
# was sampled from -1 to 1 in both directions even though I scaled the max/min
# radius of the RBFs. This needs to be checked again though as I found an error
# in my implementation and did not recheck yet

# SP_vel.clustering([8, 70], r_mM=[0.013, 0.2], eps_l=0.88) # 2097, -0.5:0.5
# SP_vel.clustering([2.5,10,30], r_mM=[0.01, 0.2], eps_l=0.6) # 1000, -1:1
# SP_vel.clustering([2.5,20], r_mM=[0.005, 0.1], eps_l=0.8) # 1000, -0.5:0.5

# Step 2. Perform the clustering
SP_vel.clustering([2.5,10], r_mM=[0.025, 0.05], eps_l=0.8) # This is the one that is currently used in the presentation


# Assemble the divergence free conditions
n_c_V = n_c_H = 50
# Left
X_Div1=np.ones(n_c_V)*(x1_hat)
Y_Div1=np.linspace(y1_hat,y2_hat,n_c_V)
# Bottom
X_Div2=np.linspace(x1_hat,x2_hat,n_c_H)
Y_Div2=np.ones(n_c_H)*y1_hat
# Right 
X_Div3=np.ones(n_c_V)*x2_hat
Y_Div3=np.linspace(y1_hat,y2_hat,n_c_V)
# Top 
X_Div4=np.linspace(x1_hat,x2_hat,n_c_H)
Y_Div4=np.ones(n_c_H)*y2_hat

# Assemble the individual arrays
X_Div=np.hstack((X_Div1,X_Div2,X_Div3,X_Div4))
Y_Div=np.hstack((Y_Div1,Y_Div2,Y_Div3,Y_Div4))

# we remove the duplicates
# TODO There is probably a better way to do this, but I did not check it in
# Pietro's code yet
duplicate_remover = pd.Series(list(map(tuple, np.column_stack((X_Div, Y_Div)))))
duplicate_remover.drop_duplicates(inplace = True, keep = 'first')
duplicate_remover = np.array(duplicate_remover.to_list())
X_Div = duplicate_remover[:,0]
Y_Div = duplicate_remover[:,1]

# We assemble the divergence free conditions
DIV = [X_Div, Y_Div]

# Step 3. Set the constraints (only divergence free for this case)

# Note: An additional RBF is located at each constraint point, as is done in 
# Pietro's implementation. @Miguel, you proposed to give this as a parameter 
# during the assembly of the poisson problem. However, I would moved it to the 
# constraint function as this is the most natural flow of the code and comes
# directly after the clustering
SP_vel.vector_constraints(DIV = DIV, extra_RBF = True)

# # These are two plots that I used to check the Radii of the RBFs during the implementation
# plt.figure()
# plt.stem(SP_vel.d_k)

# plt.figure()
# plt.stem(SP_vel.c_k)

# Plot the RBFs to check the clustering
SP_vel.plot_RBFs_2D()

# Step 4. Assemble the linear system.

# For now, I am not using the harmonic basis.
# The reason is that I did not analyze its influence yet and thus did not want to 
# put it there for the PET yet. Instead, I explored stronger penalties of the 
# divergence-free flow, in particular when the seeding is low
SP_vel.Assembly_Regression(n_hb = 0, alpha_div = 10)

# Step 5. Solve the linear system.

# The condition number that we give is relatively high but I did not find out
# yet the trade-off between conditioning and accuracy. Also important is that only
# A is regularized with a fixed condition number. The reason is that M usually 
# has a much better condition number. For the given configuration settings,
# we have 1e18 for A and only 1e9 for M. I am using the conditioning from 
# the paper but it barely affects the condition number
SP_vel.Solve(K_cond=1e8)

# Step 6. Get the solution on the scattered data points again
solution_P = SP_vel.Get_Sol(grid = [X,Y])
U_P = solution_P[:n_p]
V_P = solution_P[n_p:]

# compute the magnitude of the RBF solution
U_magn_fit = np.sqrt(U_P**2 + V_P**2)
# compute the magnitude of the analytical solution
U_magn_corr = np.sqrt(U**2 + V**2)
# compute the error in the magnitude
error_magn = np.linalg.norm(U_magn_fit - U_magn_corr) / np.linalg.norm(U_magn_corr)
# print it
print('Error total: ' + str(error_magn))

#%%
# =============================================================================
# This is where the pressure computation starts
# =============================================================================

# First, we evaluate the source term (i.e. the r.h.s. in eq. (21)).
# The function is newly added
source_term = SP_vel.Evaluate_Source_Term(grid = [X, Y], rho = rho)

# compute the pressure in a single location to have one Dirichlet condition
X_pres = -0.5; Y_pres = -0.5
radius_pres = np.sqrt(X_pres**2+Y_pres**2)
u_theta_pres = Gamma/(2*np.pi*radius_pres)*(1-np.exp(-radius_pres**2/cTH))
pres_dir = -np.array([0.5*rho*(u_theta_pres)**2-\
                rho*Gamma**2/(4*np.pi**2*cTH)*(sc.exp1(radius_pres**2/cTH)-sc.exp1(2*radius_pres**2/cTH))])

# Assemble the Neumann pressure boundary conditions (these are the same points 
# as for the divergence). I just reassemble the arrays here because I was testing
# what happens when the divergence free constraints do not coincide with the 
# neumann constraints in the pressure
n_c_V = n_c_H = 50
# Left
X_Pres_N1=np.ones(n_c_V)*(x1_hat) # we remove the point where our Dirichlet condition is
Y_Pres_N1=np.linspace(y1_hat,y2_hat,n_c_V) # we remove the point where our Dirichlet condition is
# Bottom
X_Pres_N2=np.linspace(x1_hat,x2_hat,n_c_H)[1:]
Y_Pres_N2=np.ones(n_c_H)[1:]*y1_hat
# Right 
X_Pres_N3=np.ones(n_c_V)*x2_hat
Y_Pres_N3=np.linspace(y1_hat,y2_hat,n_c_V)
# Top 
X_Pres_N4=np.linspace(x1_hat,x2_hat,n_c_H)
Y_Pres_N4=np.ones(n_c_H)*y2_hat

# Assemble the individual arrays
X_Pres_N=np.hstack((X_Pres_N1,X_Pres_N2,X_Pres_N3,X_Pres_N4))
Y_Pres_N=np.hstack((Y_Pres_N1,Y_Pres_N2,Y_Pres_N3,Y_Pres_N4)) 

# Remove the duplicates again
duplicate_remover = pd.Series(list(map(tuple, np.column_stack((X_Pres_N, Y_Pres_N)))))
duplicate_remover.drop_duplicates(inplace = True, keep = 'first')
duplicate_remover = np.array(duplicate_remover.to_list())
X_Pres_N = duplicate_remover[1:,0]
Y_Pres_N = duplicate_remover[1:,1]

# we assemble the normals in the same way
# Left
n_x_1 = np.ones(n_c_V)[:-1]*(-1)
n_y_1 = np.ones(n_c_V)[:-1]*0
# Bottom
n_x_2 = np.ones(n_c_H)[1:-1]*0
n_y_2 = np.ones(n_c_H)[1:-1]*(-1)
# Right
n_x_3 = np.ones(n_c_V)[:-1]*1
n_y_3 = np.ones(n_c_V)[:-1]*0
# Top
n_x_4 = np.ones(n_c_H)[:-1]*0
n_y_4 = np.ones(n_c_H)[:-1]*(1)

# Assemble the individual arrays
n_x = np.hstack((n_x_1, n_x_2, n_x_3, n_x_4))
n_y = np.hstack((n_y_1, n_y_2, n_y_3, n_y_4))

# We evaluate the pressure normal for the boundary conditions from the Velocity
# SPICY. Afterwards, it can technically be deleted if we are hitting memory limitations
# The function is also new
P_Neu = SP_vel.Get_Pressure_Neumann(grid = [X_Pres_N, Y_Pres_N], normals = [n_x, n_y],
                                    rho = rho, mu = mu)

# We assemble our Neumann and Dirichlet B.C.
NEU_P = [X_Pres_N, Y_Pres_N, n_x, n_y, P_Neu]
DIR_P = [np.array([X_pres]), np.array([Y_pres]), pres_dir]

# 1.: Define the class
SP_pres = spicy([np.zeros(X.shape)], [X, Y], basis = basis)

# 2.: Perform the clustering
# We take the same one as for the velocity regression, I did not yet test
# what the benefits of varying the clustering might be
SP_pres.clustering([2.5,10], r_mM=[0.025, 0.05], eps_l=0.8)

# 3.: Set boundary conditions
SP_pres.scalar_constraints(DIR = DIR_P, NEU = NEU_P, extra_RBF = True)

# Plot the RBFs to check the clustering
SP_pres.plot_RBFs_2D()

# 4. Assemble the linear system: 
SP_pres.Assembly_Poisson_2D(source_terms = source_term, n_hb = 0)

# 5. Solve the system

# The condition number is higher than for the velocity regression. The reason being
# that the Poisson problem is more illposed, the condition number is four OOM above that 
# of the velocity regression. Smaller condition numbers result in a worse solution
# as expected. However, M still has a conditioning of 1e9 so we do not fix its 
# condition number
SP_pres.Solve(K_cond = 1e12)

# We compute the pressure on the scattered points
P_calc = SP_pres.Get_Sol(grid = [X, Y])

# Compute the analytical pressure field
u_theta = Gamma/(2*np.pi*r)*(1-np.exp(-r**2/cTH))
P_corr = -0.5*rho*u_theta**2-rho*Gamma**2/(4*np.pi**2*cTH)*(sc.exp1(r**2/cTH)-sc.exp1(2*r**2/cTH))

# # Plot the resulting pressure field if desired
# plt.figure()
# cont = plt.scatter(X, Y, c = P_corr)
# plt.colorbar(cont)

# # Plot the pressure error if desired
# plt.figure()
# cont = plt.scatter(X, Y, c = P_calc - P_corr)
# plt.colorbar(cont)

# print the pressure error
error_p = np.linalg.norm(P_calc-P_corr)/np.linalg.norm(P_corr)
print('Error total, pressure: ' + str(error_p))
