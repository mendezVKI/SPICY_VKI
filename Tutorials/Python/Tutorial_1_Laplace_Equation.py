import numpy as np
import matplotlib.pyplot as plt
from spicy_vki.spicy import Spicy


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
U = X**2-Y**2

# Here we plot the solution on the sampled point (we will use it only to verify SPICY's accuracy)
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
cont = ax.scatter(X, Y, c=U)
ax.set_aspect('equal')
fig.colorbar(cont)

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

SP = Spicy([np.zeros(X.shape)], [X,Y], basis='c4', model='scalar', verbose=0)
SP.collocation([6, 50], r_mM=[0.02, 1.4], eps_l=0.88, method='clustering')
SP.scalar_constraints(DIR=DIR, extra_RBF=True)
SP.plot_RBFs(level=0) # plot the clustering results at level 0
SP.plot_RBFs(level=1) # plot the clustering results at level 1

SP.Assembly_Poisson()
SP.Solve(K_cond=1e8)
U_calc = SP.get_sol([X, Y], order=0)

error = np.linalg.norm(U_calc - U) / np.linalg.norm(U)
print('l2 relative error in  phi: {0:.3f}%'.format(error*100))

fig, axes = plt.subplots(ncols=3, figsize=(15, 5), dpi=100)
axes[0].set_title('RBF Regression')
sc = axes[0].scatter(X, Y, c=U_calc)
fig.colorbar(sc, ax=axes[0])
axes[1].set_title('Ground truth')
sc2 = axes[1].scatter(X, Y, c=U)
fig.colorbar(sc2, ax=axes[1])
axes[2].set_title('Difference')
sc3 = axes[2].scatter(X, Y, c=U_calc-U)
fig.colorbar(sc3, ax=axes[2])

for ax in axes.flatten():
    ax.set_aspect(1)
fig.tight_layout()


# We first define a new set of points. For the sake of demonstration we take a unfiform grid.
X_g, Y_g = np.meshgrid(np.linspace(x1, x2, 10), np.linspace(y1, y2, 10))

# The gradient field u=dphidx and v=dphidy should be:
u_T = 2*X_g
v_T = -2*Y_g

# Using the function Get_first_Derivatives in the scipy object we can assign the output gradient to a field:
u_C, v_C = SP.Get_first_Derivatives([X_g.ravel(), Y_g.ravel()])
# Note that the input grid should be a list

# We can now plot a quiver of the theoretical field in black and the computed one in red.
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.quiver(X_g, Y_g, u_T, v_T, color='black')
ax.quiver(X_g.ravel(), Y_g.ravel(), u_C, v_C, color='red')
fig.tight_layout()

plt.show()
