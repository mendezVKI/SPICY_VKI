# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 11:13:04 2023

@author: ManuelRatz
"""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import scipy.io
import os
from spicy_class_2103 import spicy

# Fix the random seed
np.random.seed(41)
n_p = 6000

# Properties of the domain and flow
R = 0.05
H = 0.41
L = 1.1
mu = 2e-2
rho = 1

# Load the matlab data from the ansys solution
data=scipy.io.loadmat('FluentSol.mat')
# Extract the x, y values
X = data['x'].reshape(-1)
Y = data['y'].reshape(-1) 
# Extract the velocities
vx = data['vx'].reshape(-1)
vy = data['vy'].reshape(-1)
P = data['p'].reshape(-1)

WALLBOOL=np.logical_and(vx==0,vy==0)
WALLBOOL=np.logical_or(WALLBOOL,X==0)
circlebool=np.logical_and(np.logical_and(vx==0,vy==0),np.logical_and(Y!=0,Y!=H))
#Extract the final dataset
X=X[np.invert(WALLBOOL)]
Y=Y[np.invert(WALLBOOL)]
P=P[np.invert(WALLBOOL)]
U=vx[np.invert(WALLBOOL)]
V=vy[np.invert(WALLBOOL)]

np.random.seed(47)
bool_arr=np.random.randint(low=0,high=len(X),size=n_p)
X=X[bool_arr]
Y=Y[bool_arr]
P=P[bool_arr]
U=U[bool_arr]
V=V[bool_arr]

q = 0.1
U_noise = U * (1 + q * np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q * np.random.uniform(-1, 1, size = V.shape))

n_p = X.shape[0]

plt.figure()
plt.scatter(X, Y, c = U)

#%%
# define the basis
basis = 'gauss'

# Step 1: Initialize the spicy class for the velocity regression
SP_vel = spicy([U_noise, V_noise], [X,Y], basis=basis)

H = 0.41
L = 1.1
R = 0.05
NC=150 #number of collocation points for the every BC (constraints)
# X and Y coordinates of upper boundary  
YCON1=H*np.ones(NC)# Y coordinate of upper boundary
XCON1=np.linspace(0,L,NC)# X coordinate of upper boundary
# X and Y coordinates of Bottom Boundary  
YCON3=np.zeros(NC-1)# Y coordinate of bottom boundary
XCON3=np.linspace(L/NC,L,NC-1)# X coordinate of bottom boundary
# X and Y coordinates of the Cylinder
alphaT=np.linspace(0,2*np.pi*(NC-1)/NC,NC-1) # angular spacing
# alphaT=np.hstack((alphaT,np.pi,np.pi/2,np.pi*3/2))
XCON5=0.2+R*np.cos(alphaT)
YCON5=0.2+R*np.sin(alphaT)
# X and Y coordinates of the Inlet
XCON4=np.hstack((np.zeros(NC)))#Pay attention another hide constraint is present 
#to ensure the constant value of the velocity before the inlet, therefore some constraint equal to the one
#of the inlet is added before the entry, that just why there is still no Neumann constraint on velocities
YCON4=np.hstack((np.linspace(0,H,NC)))
#At the outlet there are only divergence free constraint
XCON2=L*np.ones(NC-2)#points where contraint are applied (only divergence free)
YCON2=np.linspace(0+H/NC,H-H/NC,NC-2)#points where contraint are applied (only divergence free)
# Stack all the Standard constraint points Points
XCON=[XCON1,XCON2,XCON3,XCON4,XCON5]#points where contraint are applied
YCON=[YCON1,YCON2,YCON3,YCON4,YCON5]#points where contraint are applied
CONu1=np.zeros(len(XCON1))#conditions (no slip and inlet) for u
CONv1=np.zeros(len(XCON1))#conditions (no slip and inlet) for v
CONu3=np.zeros(len(XCON3))#conditions (no slip and inlet) for u
CONv3=np.zeros(len(XCON3))#conditions (no slip and inlet) for v
CONu4=4*(H-YCON4)*YCON4/H**2
CONv4=np.zeros(len(XCON4))
CONu5=np.zeros(len(XCON5))#conditions (no slip and inlet) for u
CONv5=np.zeros(len(XCON5))#conditions (no slip and inlet) for v
CONU=[CONu1,'only_div',CONu3,CONu4,CONu5]#Definition of constraint of velocity list
CONV=[CONv1,'only_div',CONv3,CONv4,CONv5]
CON=[CONU,CONV]#This is the input accepted by the code

# check only the divergence free constraint
CONU = ['only_div']
CONV = ['only_div']
XCON = [XCON2]
YCON = [YCON2]

# check the inlet
CONU = [CONu4]
CONV = [CONv4]
XCON = [XCON4]
YCON = [YCON4]

# We assemble the velocity constraints for Dirichlet and Divergence-free flow
DIR = [XCON5, YCON5, CONu5, CONu5]
DIV = [XCON5, YCON5]

X_Div = np.concatenate((XCON1, XCON2, XCON3, XCON4, XCON5))
Y_Div = np.concatenate((YCON1, YCON2, YCON3, YCON4, YCON5))

X_Dir = np.concatenate((XCON1, XCON3, XCON4, XCON5))
Y_Dir = np.concatenate((YCON1, YCON3, YCON4, YCON5))
U_Dir = np.concatenate((CONu1, CONu3, CONu4, CONu5))
V_Dir = np.concatenate((CONv1, CONv3, CONv4, CONv5))

_, valid_idcs = np.unique(np.column_stack((X_Div, Y_Div)),
                          return_index = True, axis = 0)
X_Div = X_Div[valid_idcs]
Y_Div = Y_Div[valid_idcs]
DIV = [X_Div, Y_Div]


_, valid_idcs = np.unique(np.column_stack((X_Dir, Y_Dir)),
                          return_index = True, axis = 0)
X_Dir = X_Dir[valid_idcs]
Y_Dir = Y_Dir[valid_idcs]
U_Dir = U_Dir[valid_idcs]
V_Dir = V_Dir[valid_idcs]
DIR = [X_Dir, Y_Dir, U_Dir, V_Dir]

# Step 2: Perform the clustering
# SP_vel.clustering([3.5,30], r_mM=[0.0025, 0.2], eps_l=0.8) # Gaussian
SP_vel.clustering([4,20], r_mM=[0.005, 0.2], eps_l=0.8)

# Step 3: Set the constraints
SP_vel.vector_constraints(DIR = DIR, DIV = DIV, extra_RBF = True)
# SP_vel.vector_constraints(DIV = DIV, extra_RBF = True)
# SP_vel.X_C = np.load(Fol_In + os.sep + 'X_C_p.npy')
# SP_vel.Y_C = np.load(Fol_In + os.sep + 'Y_C_p.npy')
# SP_vel.c_k = np.load(Fol_In + os.sep + 'c_k_p.npy')


# plt.figure()
# plt.scatter(SP_vel.X_C, SP_vel.Y_C, c = SP_vel.c_k)

# Plot the RBFs to check the clustering
SP_vel.plot_RBFs()

#%%

# Step 4. Assemble the linear system
SP_vel.Assembly_Regression(n_hb = 0, alpha_div = 1)

# Step 5. Solve the linear system
SP_vel.Solve(K_cond=1e12)

# Step 6. Get the solution on the scattered data points again
solution_P = SP_vel.Get_Sol(grid = [X,Y])
U_P = solution_P[:n_p]
V_P = solution_P[n_p:]

fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (15, 5), dpi = 100)
cont = ax[0].scatter(X, Y, c = U)
cont = ax[1].scatter(X, Y, c = U_P)
cont = ax[2].scatter(X, Y, c = np.abs(U_P - U))
# fig.colorbar(cont)

fig, ax = plt.subplots(ncols = 3, nrows = 1, figsize = (15, 5), dpi = 100)
cont = ax[0].scatter(X, Y, c = V)
cont = ax[1].scatter(X, Y, c = V_P)
cont = ax[2].scatter(X, Y, c = np.abs(V_P - V))


# compute the magnitude of the RBF solution
U_magn_fit = np.sqrt(U_P**2 + V_P**2)
# compute the magnitude of the analytical solution
U_magn_corr = np.sqrt(U**2 + V**2)
# compute the error in the magnitude
error_magn = np.linalg.norm(U_magn_fit - U_magn_corr) / np.linalg.norm(U_magn_corr)
# print it
print('Error total: ' + str(error_magn))

source_term = SP_vel.Evaluate_Source_Term(grid = [X, Y], rho = rho)

#%%

mu = 2e-2
rho = 1

XBC1=np.linspace(0.0,L,NC)#X position of BC point in the lower edge
YBC1=np.zeros(NC)#Y position of BC point in the lower edge
XBC2=L*np.ones(NC)#X position of BC point in the right edge
YBC2=np.linspace(0,H,NC)#Y position of BC point in the right edge
XBC3=np.linspace(0,L,NC)#X position of BC point in the upper edge
YBC3=H*np.ones(NC)#Y position of BC point in the upper edge
XBC4=0.0*np.ones(NC)#X position of BC point in the left edge
YBC4=np.linspace(0,H,NC)#X position of BC cons point in the left edge
alphaP=np.linspace(0,2*np.pi*(NC-1)/NC,NC) 
 
XBC5=0.2+R*np.cos(alphaP)#X position of BC point in the cylinder
YBC5=0.2+R*np.sin(alphaP)#Y position of BC point in the cylinder

# print(XBC5)
# print(YBC5)

#Array for normal direction to the walls
n1=np.vstack((np.zeros(len(XBC1)),-np.ones(len(XBC1))))
n2=np.vstack((np.ones(len(XBC2)),np.zeros(len(XBC2))))
n3=np.vstack((np.zeros(len(XBC3)),np.ones(len(XBC3))))
n4=np.vstack((-np.ones(len(XBC2)),np.zeros(len(XBC2))))
n5=np.vstack((-np.cos(alphaP),-np.sin(alphaP)))

X_Pres_N = np.concatenate((XBC1, XBC3, XBC4, XBC5))
Y_Pres_N = np.concatenate((YBC1, YBC3, YBC4, YBC5))
n_x = np.concatenate((n1[0,:], n3[0,:], n4[0,:], n5[0,:]))
n_y = np.concatenate((n1[1,:], n3[1,:], n4[1,:], n5[1,:]))


# X_Pres_N = XBC5
# Y_Pres_N = YBC5
# n_x = n5[0,:]
# n_y = n5[1,:]


# _, valid_idcs = np.unique(np.column_stack((X_Pres_N, Y_Pres_N)),
#                           return_index = True, axis = 0)
# X_Pres_N = X_Pres_N[valid_idcs]
# Y_Pres_N = Y_Pres_N[valid_idcs]
# n_x = n_x[valid_idcs]
# n_y = n_y[valid_idcs]
# # plt.figure(dpi = 200)
# # plt.quiver(X_Pres_N, Y_Pres_N, n_x, n_y)

P_Neu = SP_vel.Get_Pressure_Neumann(
    grid = [X_Pres_N, Y_Pres_N], normals = [n_x, n_y], rho = rho, mu = mu)
# plt.figure()
# # cont = plt.scatter(X_Pres_N1, Y_Pres_N1, c = P_Neu)
# cont = plt.scatter(X_Pres_N, Y_Pres_N, c = P_Neu)
# plt.colorbar(cont)

X_Pres_D = XBC2
Y_Pres_D = YBC2

NEU_P = [X_Pres_N, Y_Pres_N, n_x, n_y, P_Neu]
DIR_P = [X_Pres_D, Y_Pres_D, np.zeros(X_Pres_D.shape)]

# 1.: Define the class
SP_pres = spicy([np.zeros(X.shape)], [X, Y], basis = basis)

# 2.: Perform the clustering
# We take the same one as for the velocity regression, I did not yet test
# what the benefits of varying the clustering might be
SP_pres.clustering([4, 20], r_mM=[0.005, 0.2], eps_l=0.8)

# 3.: Set boundary conditions
SP_pres.scalar_constraints(DIR = DIR_P, NEU = NEU_P, extra_RBF = True)
# SP_pres.scalar_constraints(NEU = NEU_P, extra_RBF = True)
# SP_pres.scalar_constraints(DIR = DIR_P, extra_RBF = True)

# SP_pres.X_C = np.load(Fol_In + os.sep + 'X_C_pres_p.npy')
# SP_pres.Y_C = np.load(Fol_In + os.sep + 'Y_C_pres_p.npy')
# SP_pres.c_k = np.load(Fol_In + os.sep + 'c_k_pres_p.npy')
# SP_pres.d_k = 1/SP_pres.c_k*np.sqrt(np.log(2))

# 4. Assemble the linear system: 
SP_pres.Assembly_Poisson(source_terms = source_term, n_hb = 0)

# 5. Solve the system

# The condition number is higher than for the velocity regression. The reason being
# that the Poisson problem is more illposed, the condition number is four OOM above that 
# of the velocity regression. Smaller condition numbers result in a worse solution
# as expected. However, M still has a conditioning of 1e9 so we do not fix its 
# condition number
SP_pres.Solve(K_cond = 1e12)

# We compute the pressure on the scattered points
P_calc = SP_pres.Get_Sol(grid = [X, Y])

SP_pres.plot_RBFs()

# print the pressure error
error_p = np.linalg.norm(P_calc-P)/np.linalg.norm(P)
print('Error total, pressure: ' + str(error_p))

# # Plot the pressure error if desired
# plt.figure()
# cont = plt.scatter(X, Y, c = p, vmin = 0, vmax = 3.6)
# plt.colorbar(cont)

plt.figure()
cont = plt.scatter(X, Y, c = P_calc - P)
plt.colorbar(cont)
