# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 15:05:27 2025

@author: rigutto
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import urllib.request, os
from zipfile import ZipFile

#%% DNS boundary layer data (test case 1)

def load_DNS_data_points(n_p, x_min, x_max, y_min, y_max):
    """
    Serial function to read the DNS snapshots
    """

    x_p = []
    y_p = []
    u_p = []
    v_p = []
    
    folder = os.path.join('data', 'DNS_snapshots', 'data_points')
    
    file_list = os.listdir(folder)
    
    n_f = len(file_list)
    
    for file in tqdm(file_list, desc='Import DNS data points'):
        
        # for each timestep we should sample the same amount of particles
        # as to be a multiple of 5000
        
        # load entire frame
        data = np.loadtxt(folder + os.sep + file)
        
        # adapt axis
        x = 1-data[:, 0]
        y = 1-data[:, 1]
        
        # take the required number of points within the domain
        ind = np.where((x > x_min) & (x < x_max) & (y > y_min) & (y < y_max))[0][:int(n_p/n_f)]
        
        # add them to the final list of points
        x_p.extend(1-data[ind, 0])
        y_p.extend(1-data[ind, 1])
        u_p.extend(data[ind, 3])
        v_p.extend(data[ind, 4])
    
    x_p = np.array(x_p)
    y_p = np.array(y_p)
    u_p = np.array(u_p)
    v_p = np.array(v_p)
    
    return x_p, y_p, u_p, v_p


# Downloading the data
print("Downloading data…")
url = "https://osf.io/4bw5z/download"
zip_name = "data.zip"

urllib.request.urlretrieve(url, zip_name)

with ZipFile(zip_name, "r") as zf:
    zf.extractall()

os.remove(zip_name)


x_min = 0.2
x_max = 0.7
y_min = 0.0
y_max = 0.7

# Number of particles used for plotting
n_p = int(5e4)
    
# DNS profiles
data_DNS = np.loadtxt(os.path.join('data', 'DNS_snapshots', 'profiles.txt'), skiprows=2)

# y+ profile
y_plus = data_DNS[:, 0]/0.0499*(5e-5)
f_plus = data_DNS[:, 1]*0.0499

# Load the DNS data (expensive)
x_p_all, y_p_all, f_p_all, _ = load_DNS_data_points(n_p=1e10, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max)

x_p = x_p_all[:n_p]
y_p = y_p_all[:n_p]
f_p = f_p_all[:n_p]

x_p_out = x_p_all[n_p:]
y_p_out = y_p_all[n_p:]
f_p_out = f_p_all[n_p:]

# Plot the ensemble of points
fig, ax = plt.subplots(figsize=(6.5, 6.5), layout='constrained')
sc = ax.scatter(x_p, y_p, c=f_p, label='PTV', s=3)
ax.set_xlabel(r'$x\ [-]$')
ax.set_ylabel(r'$y\ [-]$')

x_slice_min, x_slice_max = 0.44, 0.46
ax.axvline(x_slice_min, color='red', linestyle='--', linewidth=1)
ax.axvline(x_slice_max, color='red', linestyle='--', linewidth=1)
ax.fill_betweenx([y_p.min(), y_p.max()], x_slice_min, x_slice_max, color='red', alpha=0.5, label='Profile slice')

cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
cbar.set_label('Sampled $u(\\mathbf{x}_k)$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal')
plt.show()

# Profile the highlighted slice

fig, ax = plt.subplots(figsize=(6.5, 4), layout='constrained')
mask = (x_p >= x_slice_min) & (x_p <= x_slice_max)
ax.plot(f_p[mask], y_p[mask], 'ko', markersize=3, label='Noisy BL Sample')
ax.plot(f_plus, y_plus, 'r-',linewidth=2)
ax.set_ylabel(r'$y\ [-]$')
ax.set_xlabel(r'$u\ [-]$')
ax.set_ylim(0,0.7)
ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()


#%% Experimental PTV jet data (test case 2)

# Load the data
folder = os.path.join('data', '3D_JET')
X = np.load(os.path.join(folder, 'X.npy'))
Y = np.load(os.path.join(folder, 'Y.npy'))
Z = np.load(os.path.join(folder, 'Z.npy'))

R = np.sqrt(Y**2 + Z**2)
ind = np.where((np.arctan2(Z,Y)>0))
R[ind] = -R[ind]
Y = R

U = np.load(os.path.join(folder, 'U.npy'))

scale = np.max(X)

X = X/(scale)
Y = Y/(scale)
Z = Z/(scale)

# Select subsample
n_p = int(5e4)

ind = np.random.choice(len(X), n_p, replace=False)

x_p = X[ind[:n_p]]
y_p = Y[ind[:n_p]]
f_p = U[ind[:n_p]]

# Plot the data ensemble
fig, ax = plt.subplots(figsize=(6.5, 4), layout='constrained')
sc = ax.scatter(x_p*scale/2, y_p*scale/2, c=f_p, label='PTV', s=3)
ax.set_xlabel(r'$x/D\ [-]$')
ax.set_ylabel(r'$y/D\ [-]$')
x_slice_min, x_slice_max = 1.4/scale*2, 1.5/scale*2
ax.axvline(x_slice_min*scale/2, color='red', linestyle='--', linewidth=1)
ax.axvline(x_slice_max*scale/2, color='red', linestyle='--', linewidth=1)
ax.fill_betweenx([y_p.min()*scale/2, y_p.max()*scale/2], x_slice_min*scale/2, x_slice_max*scale/2, color='red', alpha=0.5, label='Profile slice')
cbar = plt.colorbar(sc, ax=ax, orientation='vertical', fraction=0.03, pad=0.04)
cbar.set_label('Sampled $u(\\mathbf{x}_k)$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_aspect('equal')
plt.show()

# Plot the slice
mask = (x_p >= x_slice_min) & (x_p <= x_slice_max)
x_slice = x_p[mask]; y_slice = y_p[mask]; f_slice = f_p[mask]

fig, ax = plt.subplots(figsize=(6.5, 4), layout='constrained')
ax.plot(f_slice, y_slice*scale/2, 'ko', markersize=3)
ax.set_xlabel(r'$u\ [-]$')
ax.set_ylabel(r'$y/D\ [-]$')
# ax.set_ylim(0,0.7)
ax.grid(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()
