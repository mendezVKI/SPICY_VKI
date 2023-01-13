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
n_p=1000
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
X_hat=(X-x1)/(x2-x1)-0.5
Y_hat=(Y-y1)/(y2-y1)-0.5

plt.plot(X_hat,Y_hat,'ko')

x1_hat,x2_hat=-0.5,0.5
y1_hat,y2_hat=-0.5,0.5




from spicy_class import spicy

SP=spicy([U,U],[X_hat,Y_hat],model='laminar',basis='gauss')

#%% Define Constraints.
# Let's use the same number of points for the constraints along the patches.
n_c_H=100 # number of constrained points
n_c_V=50 # number of constrained points

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
c_D3=np.sin(2*np.pi*YCON3)

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


SP.clustering([5,30],r_mM=[0.01,0.2],eps_l=0.7)


fig, ax = plt.subplots(figsize=(8, 4)) 
plt.axis('equal')
for i in range(0,len(SP.X_C),1):
    circle1 = plt.Circle((SP.X_C[i], SP.Y_C[i]), SP.d_k[i]/2, 
                         fill=True,color='g',edgecolor='k',alpha=0.2)
    ax.add_artist(circle1)
plt.scatter(SP.XG,SP.YG,c=SP.u)
plt.plot(XCON,YCON,'ro')

plt.show()

plt.figure(2)
plt.plot(SP.d_k/2,'ko')

#%% in case of Neuman conditions.

# # For debugging purposes, this is the case in which Neuman conditions
# # are set on the left boundary XCON1:
    
# n_x1=-np.ones(len(XCON1))
# n_y1=np.zeros(len(XCON1))
# c_N1=np.zeros(len(XCON1))

# # Dirichlet conditions
# DIR=[np.hstack((XCON2,XCON3,XCON4)),
#      np.hstack((YCON2,YCON3,YCON4)),
#      np.hstack((c_D2,c_D3,c_D4))]

# # Neuman conditions
# NEU=[XCON1,YCON1,n_x1,n_y1,c_N1]

# SP.scalar_constraints(DIR,NEU)














