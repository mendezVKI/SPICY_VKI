# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 15:28:47 2023

@author: admin
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
X = X[inlet_and_wall_remover]; Y = Y[inlet_and_wall_remover]; P = P[inlet_and_wall_remover]
U = U[inlet_and_wall_remover]; V = V[inlet_and_wall_remover]

# From the remaining points we can choose to sample a random amount if we want to go for a smaller test case. In this
# tutorial, we take the maximum number of points which is 18755
n_p = 18755
random_points_indices = np.random.randint(low=0, high=len(X), size=n_p)
# Select the data points
X = X[random_points_indices]; Y = Y[random_points_indices]; P = P[random_points_indices]
U = U[random_points_indices]; V = V[random_points_indices]

# Add 0.3 noise to the velocity field
q = 0.05
U_noise = U * (1 + q * np.random.uniform(-1, 1, size = U.shape))
V_noise = V * (1 + q * np.random.uniform(-1, 1, size = V.shape))


plt.scatter(X,Y,c=P)

# Define polygons as follows
from shapely import geometry

p1 = geometry.Point(0.1,0.1)
p2 = geometry.Point(0.3,0.1)
p3 = geometry.Point(0.3,0.3)
p4 = geometry.Point(0.1,0.3)

pointList = [p1, p2, p3, p4]
poly1 = geometry.Polygon([i for i in pointList])

plt.plot(*poly1.exterior.xy,'r')

# Define also a second box

p1 = geometry.Point(0.05,0.05)
p2 = geometry.Point(0.45,0.05)
p3 = geometry.Point(0.45,0.35)
p4 = geometry.Point(0.05,0.35)

pointList = [p1, p2, p3, p4]
poly2 = geometry.Polygon([i for i in pointList])

plt.plot(*poly2.exterior.xy,'r')


# Find the points INSIDE poly 1:
List_1=[]

for j in range(len(X)):
 List_1.append(poly1.contains(geometry.Point(X[j],Y[j])))


plt.plot(X[List_1],Y[List_1],'ro')


# The list of polygon could be 
List_Poly=[poly1,poly2,[]]


SP_U = spicy([U_noise], [X,Y], basis='c4') # initialize object
SP_U.clustering([6,20,100], Areas=[[],[],[]], r_mM=[0.05,0.6], eps_l=0.88) # cluster


# We set the constraints in these points and also place additional RBFs in each of these points
SP_U.scalar_constraints()

# Plot the RBFs and the clusters
SP_U.plot_RBFs(l=0)












