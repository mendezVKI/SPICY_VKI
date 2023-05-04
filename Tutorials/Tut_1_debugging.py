# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 18:34:47 2023

@author: ratz, mendez
"""


import numpy as np
import matplotlib.pyplot as plt 
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
n_p = 400

# Define the domain boundaries
x1, x2 = 0, 1 
y1, y2 = 0, 1 

# Generate the random points (note: we write the code for sampling on an arbitrary rectangular domain)
X = np.random.random(n_p)*(x2 - x1) + x1
Y = np.random.random(n_p)*(y2 - y1) + y1

# The analytical solution for this is the following
U=X**2-Y**2

P=plt.scatter(X,Y,c=U); plt.gca().set_aspect('equal')
plt.colorbar(P)



#%% We use the RBF regression

# Define the boundary conditions
# Number of points for the vertical and horizontal boundary
n_c_V = n_c_H = 10

# Left boundary (x=x1, y=line)
X_Dir1 = np.ones(n_c_V)*(x1)
Y_Dir1 = np.linspace(y1,y2,n_c_V)
U_Dir1 = -Y_Dir1**2
# Bottom boundary (x=line, y=y1)
X_Dir2 = np.linspace(x1,x2,n_c_H)
Y_Dir2 = np.ones(n_c_H)*y1
U_Dir2 = X_Dir2**2
# Right boundary (x=x2, y=line)
X_Dir3 = np.ones(n_c_V)*x2
Y_Dir3 = np.linspace(y1,y2,n_c_V)
U_Dir3 = 1-Y_Dir3**2
# Top  boundary
X_Dir4 = np.linspace(x1,x2,n_c_H)
Y_Dir4 = np.ones(n_c_H)*y2
U_Dir4 = X_Dir4**2-1


# Assemble the constraints
X_Dir = np.concatenate((X_Dir1, X_Dir2, X_Dir3, X_Dir4))
Y_Dir = np.concatenate((Y_Dir1, Y_Dir2, Y_Dir3, Y_Dir4))
U_Dir = np.concatenate((U_Dir1, U_Dir2, U_Dir3, U_Dir4))
DIR = [X_Dir, Y_Dir, U_Dir]

# We set the constraints in these points and also place additional RBFs in each of these points
SP = spicy([np.zeros(X.shape)], [X,Y], basis='c4')
SP.clustering([6,50], Areas=[[],[]], r_mM=[0.01,0.7], eps_l=0.88)
SP.scalar_constraints(DIR=DIR, extra_RBF=True)

# Plot the RBFs and the clusters
SP.plot_RBFs(l=0)
SP.plot_RBFs(l=1)


# Assembly Solver
SP.Assembly_Poisson()

# Solve the System
SP.Solve(K_cond=1e10)

# Get solution
U_calc = SP.Get_Sol([X,Y])

error = np.linalg.norm(U_calc - U) / np.linalg.norm(U)
print('l2 relative error in  phi: {0:.3f}%'.format(error*100))

fig, axes = plt.subplots(ncols=3, figsize=(15,5), dpi=100)
axes[0].set_title('RBF Regression')
sc=axes[0].scatter(X, Y, c=U_calc)
plt.colorbar(sc)
axes[1].set_title('Ground truth')
sc2=axes[1].scatter(X, Y, c=U)
plt.colorbar(sc2)
axes[2].set_title('Difference')
sc3=axes[2].scatter(X, Y, c=U_calc-U)
plt.colorbar(sc3)

for ax in axes.flatten():
    ax.set_aspect(1)
fig.tight_layout()


#%% Test the derivatives

# We look for derivatives in a new grid.
Xg, Yg = np.meshgrid(np.linspace(x1,x2,10), 
                     np.linspace(y1,y2,10))

# The true velocity field  is
u_T=2*Xg
v_T=-2*Yg

# Now the computed one will be
u_C,v_C=SP.Get_first_Derivatives([Xg.reshape(-1),
                                  Yg.reshape(-1)])    

plt.figure()
plt.quiver(Xg,Yg,u_T,v_T,color='blue')

plt.quiver(Xg.reshape(-1),Yg.reshape(-1),
           u_C,v_C,color='red')








