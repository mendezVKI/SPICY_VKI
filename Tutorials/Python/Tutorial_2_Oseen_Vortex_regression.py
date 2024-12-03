import numpy as np
import matplotlib.pyplot as plt
from spicy_vki.spicy import Spicy

# This is for plot customization
fontsize = 12
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
x1_hat, x2_hat = -0.5, 0.5  # m, m
y1_hat, y2_hat = -0.5, 0.5  # m, m
rho = 1  # kg/m^3
mu = 0  # Pa s

# Generate the random points
X = np.random.random(n_p) * (x2_hat - x1_hat) + x1_hat
Y = np.random.random(n_p) * (y2_hat - y1_hat) + y1_hat

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

# Add 0.4 max noise to it
q = 0.4
U_noise = U * (1 + q*np.random.uniform(-1, 1, size=U.shape))
V_noise = V * (1 + q*np.random.uniform(-1, 1, size=V.shape))

# Plot the sampled field: black arrows for the true fields, blu arrows for the noisy one
fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
ax.quiver(X, Y, U, V, np.sqrt(U**2 + V**2))
ax.quiver(X, Y, U_noise, V_noise)
ax.set_aspect('equal')
fig.tight_layout()

# We use one regression for each component.
SP_U = Spicy([U_noise], [X, Y], basis='c4', model='scalar') # initialize object
SP_U.collocation([4, 10], Areas=None, r_mM=[0.1, 1.2], eps_l=0.87)  # cluster
SP_U.scalar_constraints()  # add no constraints!
SP_U.plot_RBFs(l=0)  # plot the result of the clustering

SP_U.Assembly_Regression()  # Assembly the linear system
SP_U.Solve(K_cond=1e8)  # Solve the regression with regularization active when condition number exceed 1e8)
U_c = SP_U.Get_Sol([X, Y])  # get solution on (X,Y)
# Evaluate the error wrt to the noise free data
error = np.linalg.norm(U_c - U) / np.linalg.norm(U)
print('l2 relative error in u component: {0:.3f}%'.format(error*100))

SP_V = SP_U  # clone the SPICY object
SP_V.u = V_noise[:, np.newaxis]  # Change only the target data (for scalar, this is u)
SP_V.Assembly_Regression()  # Assembly the linear system
SP_V.Solve(K_cond=1e8)  # Solve the regression with regularization active when condition number exceed 1e8)
V_c = SP_V.Get_Sol([X, Y])  # get solution on (X,Y)
# Evaluate the error wrt to the noise free data
error = np.linalg.norm(V_c - V) / np.linalg.norm(V)
print('l2 relative error in v component: {0:.3f}%'.format(error*100))

# Then plot the results:
fig, ax = plt.subplots(figsize=(7, 5), dpi=100)
ax.quiver(X, Y, U, V)
ax.quiver(X, Y, U_noise, V_noise, color='blue')
ax.quiver(X, Y, U_c, V_c, color='red')
ax.set_aspect('equal')

SP_vec = Spicy([U_noise, V_noise], [X, Y], basis='c4', model='laminar')  # create SPICY object
# clone the cluster data:
SP_vec.r_mM = SP_V.r_mM
SP_vec.eps_l = SP_V.eps_l
SP_vec.X_C = SP_V.X_C
SP_vec.Y_C = SP_V.Y_C
SP_vec.c_k = SP_V.c_k
SP_vec.d_k = SP_V.d_k
# Proceed as usual: constraints + assembly regression + solve
SP_vec.vector_constraints()  # add no constraints!
SP_vec.Assembly_Regression(alpha_div=1)  # assembly with a penalty of 1 on div
SP_vec.Solve(K_cond=1e8)  # Solve as usual
# Get the results on the same grid:
U_c, V_c = SP_vec.Get_Sol([X, Y])

error = np.linalg.norm(U_c - U) / np.linalg.norm(U)
print('l2 relative error in u component: {0:.3f}%'.format(error*100))
error = np.linalg.norm(V_c - V) / np.linalg.norm(V)
print('l2 relative error in v component: {0:.3f}%'.format(error*100))

# We look for derivatives in a new grid.
X_g, Y_g = np.meshgrid(np.linspace(-0.5, 0.5, 100), np.linspace(-0.5, 0.5, 100))


# Derivative calculations for the unconstrained case
dudx, _ = SP_U.get_sol([X_g.ravel(), Y_g.ravel()], order=1)
_, dvdy = SP_V.get_sol([X_g.ravel(), Y_g.ravel()], order=1)

DIV = dudx + dvdy


# Derivative calculations for penalized case
dudx_p, _, _, dvdy_p = SP_vec.get_sol([X_g.ravel(), Y_g.ravel()], order=1)
DIV_p = dudx_p + dvdy_p


fig, axes = plt.subplots(ncols=2, figsize=(15,5), dpi=100)
axes[0].set_title('Free Case')
sc = axes[0].scatter(X_g, Y_g, c=DIV)
plt.colorbar(sc, ax=axes[0])
axes[1].set_title('Penalized Case')
sc2 = axes[1].scatter(X_g, Y_g, c=DIV_p)
plt.colorbar(sc2, ax=axes[1])

for ax in axes.flatten():
    ax.set_aspect(1)
fig.tight_layout()


plt.show()
