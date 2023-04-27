# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:39:31 2023

@author: mendez
"""

#%% Debugging of clustering + RBF plotting


# This tests and debug the clustering approach in 2D and 3D, together with
# the plotting utilities.

# Test 1: We consider the domain x\in[0,2] and y\in[0,1] and randomly place n_p particles.
# Then test the clustering for various levels.

# Test 2: Same as before, but for the domain x\in[0,2], y\in[0,1] and z\in[-1,1].

# In both cases, we then plot the exp and c4 collocation.



import numpy as np
import matplotlib.pyplot as plt 

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

# In order to let the polynomial basis in the Phi
# do a good job, you better position the domain differently.
X_hat=2*(X-x1)/(x2-x1)-1
Y_hat=2*(Y-y1)/(y2-y1)-1

# plt.plot(X_hat,Y_hat,'ko')

x1_hat,x2_hat=-1,1
y1_hat,y2_hat=-1,1


from spicy_class import spicy

SP=spicy([U],[X_hat,Y_hat],model='scalar',basis='gauss')

SP.clustering([5,30],r_mM=[0.01,0.2],eps_l=0.7)

SP.plot_RBFs_2D()



#%% Define Constraints.
# Let's use the same number of points for the constraints along the patches.
n_c_H=50 # number of constrained points
n_c_V=100 # number of constrained points

# this problem has only Dirichlet BC. Hence we have:

# Left
XCON1=np.ones(n_c_V)*(x1_hat)
YCON1=np.linspace(y1_hat,y2_hat,n_c_V)
c_D1=np.zeros(n_c_V)

# Bottom
XCON2=np.linspace(x1_hat,x2_hat,n_c_H)
YCON2=np.ones(n_c_H)*y1_hat
c_D2=np.zeros(n_c_H)

# Right 
XCON3=np.ones(n_c_V)*x2_hat
YCON3=np.linspace(y1_hat,y2_hat,n_c_V)
c_D3=np.sin(2*np.pi*(YCON3+1)/2) # Be careful about the change of variables

# Top 
XCON4=np.linspace(x1_hat,x2_hat,n_c_H)
YCON4=np.ones(n_c_H)*y2_hat
c_D4=np.zeros(n_c_H)

# Put together in lists :
# XCON=[XCON1,XCON2,XCON3,XCON4]
# YCON=[YCON1,YCON2,YCON3,YCON4]
# UCON=[UCON1,UCON2,UCON3,UCON4]

# or.... we directly set it as arrays:    
# This would create a vector:    
XCON=np.hstack((XCON1,XCON2,XCON3,XCON4))
YCON=np.hstack((YCON1,YCON2,YCON3,YCON4))
c_D=np.hstack((c_D1,c_D2,c_D3,c_D4))
DIR=[XCON,YCON,c_D]

# For this problem, the number of constrains (Dirichlet, Neuman) is


# Set the constraints
SP.scalar_constraints(DIR)


SP.clustering([3,10],r_mM=[0.01,0.1],eps_l=0.3)

SP.plot_RBFs_2D()


SP.Assembly_Poisson_2D(source_terms=np.zeros_like(SP.XG),n_hb=1)

SP.Solve_2D(K_cond=1e1)

U_P=SP.Get_Sol_2D(X_hat,Y_hat)

error=np.linalg.norm(U_P-U)
print(error)

# plot the solution and the error
fig, axs = plt.subplots(1, 2)
axs[0].scatter(X_hat,Y_hat,c=U)
axs[0].set_title('Provided Data')
axs[0].set_xlim([-1,1])
axs[1].scatter(X_hat,Y_hat,c=U_P)
axs[1].set_title('RBF Reconstruction')
axs[1].set_xlim([-1,1])

# from spicy_class import Phi_2D_harm, Phi_2D_RBF

# # check to see what is wrong
# W_P=np.linalg.solve(np.vstack((np.hstack((SP.A,SP.B)),
#                                 np.hstack((SP.B.T,np.zeros((len(SP.b_2),len(SP.b_2))))))),
#                                 np.hstack((SP.b_1,SP.b_2)))
# w=W_P[:len(SP.b_1):]
# Phi=np.hstack((Phi_2D_harm(SP.XG, SP.YG, SP.n_hb),
#                 Phi_2D_RBF(SP.XG, SP.YG, SP.X_C, SP.Y_C, SP.c_k)))  
# U_n=Phi.dot(w)

# plt.scatter(X_hat,Y_hat,c=U_n)


#%% In case of Neuman conditions.

# # For debugging purposes, this is the case in which Neuman conditions
# # are set on the left boundary XCON1:
    
# n_x1=-np.ones(len(XCON1))
# n_y1=np.zeros(len(XCON1))
# c_N1=np.zeros(len(XCON1))

# # Dirichlet conditions
# DIR=[np.hstack((XCON2,XCON3,XCON4)),
#       np.hstack((YCON2,YCON3,YCON4)),
#       np.hstack((c_D2,c_D3,c_D4))]

# # Neuman conditions
# NEU=[XCON1,YCON1,n_x1,n_y1,c_N1]

# SP.scalar_constraints(DIR,NEU)














