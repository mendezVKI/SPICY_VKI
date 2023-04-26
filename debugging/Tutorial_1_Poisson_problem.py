# -*- coding: utf-8 -*-
"""
Created on Wed Mar 1 13:39:31 2023

@author: ratz
"""

import numpy as np
import matplotlib.pyplot as plt 
from spicy_class_2103 import spicy

# number of particles
n_p=5000
# This would be the sampling in the real domain
rnd = np.random.default_rng(seed=39)
x1,x2=0,2
y1,y2=0,1
X=rnd.random(n_p)*(x2-x1)+x1
Y=rnd.random(n_p)*(y2-y1)+y1

# The value of the function would be:
U=np.sin(2*np.pi*Y)*(np.sinh(2*np.pi*X))/(np.sinh(4*np.pi))    
U_noise = np.zeros(X.shape)
U_noise = U

X=2*(X-x1)/(x2-x1)-1
Y=2*(Y-y1)/(y2-y1)-1

x1_hat,x2_hat=-1,1
y1_hat,y2_hat=-1,1

SP=spicy([U_noise],[X,Y],basis='gauss')

r_m, r_M = 0.01, 0.6
SP.clustering([30],r_mM=[r_m,r_M],eps_l=0.8)

# Number of points for the vertical and horizontal boundary
n_c_V = n_c_H = 50

# Left boundary
X_Dir1 = np.ones(n_c_V)*(x1_hat)
Y_Dir1 = np.linspace(y1_hat,y2_hat,n_c_V)
U_Dir1 = np.zeros(n_c_V)
# Bottom boundary
X_Dir2 = np.linspace(x1_hat,x2_hat,n_c_H)
Y_Dir2 = np.ones(n_c_H)*y1_hat
U_Dir2 = np.zeros(n_c_H)
# Right boundary
X_Dir3 = np.ones(n_c_V)*x2_hat
Y_Dir3 = np.linspace(y1_hat,y2_hat,n_c_V)
U_Dir3 = np.sin(2*np.pi*(Y_Dir3+1)/2) # Be careful about the change of variables
# Top  boundary
X_Dir4 = np.linspace(x1_hat,x2_hat,n_c_H)
Y_Dir4 = np.ones(n_c_H)*y2_hat
U_Dir4 = np.zeros(n_c_H)

# Assemble the constraints
X_Dir = np.concatenate((X_Dir1, X_Dir2, X_Dir3, X_Dir4))
Y_Dir = np.concatenate((Y_Dir1, Y_Dir2, Y_Dir3, Y_Dir4))
U_Dir = np.concatenate((U_Dir1, U_Dir2, U_Dir3, U_Dir4))
DIR = [X_Dir, Y_Dir, U_Dir]

# We set the constraints in these points and also place additional RBFs in each of these points
SP.scalar_constraints(DIR=DIR, extra_RBF=True)


SP.plot_RBFs()

SP.Assembly_Poisson(n_hb=8)

SP.Solve(K_cond=1e12)

U_P=SP.Get_Sol([X,Y])

error=np.linalg.norm(U_P-U) / np.linalg.norm(U)
print(error)

# plot the solution and the error
# fig, axes = plt.subplots(ncols=2, figsize=(10,5), dpi=100)
# axes[0].scatter(X,Y,c=U)
# axes[0].set_title('Provided Data')
# axes[0].set_xlim([-1,1])
# axes[1].scatter(X,Y,c=U_P)
# axes[1].set_title('RBF Reconstruction')
# axes[1].set_xlim([-1,1])

# fig, axes = plt.subplots(ncols=3, figsize=(15,5), dpi=100)
# axes[0].scatter(X, Y, c=U)
# axes[1].scatter(X, Y, c=U_P)
# axes[2].scatter(X, Y, c=U-U_P)

plt.figure(figsize=(6,5), dpi=100)
plt.scatter(X, Y, c=U)
plt.colorbar()

plt.figure(figsize=(6,5), dpi=100)
plt.scatter(X, Y, c=U_P)
plt.colorbar()

plt.figure(figsize=(6,5), dpi=100)
plt.scatter(X, Y, c=U_P-U)
plt.colorbar()