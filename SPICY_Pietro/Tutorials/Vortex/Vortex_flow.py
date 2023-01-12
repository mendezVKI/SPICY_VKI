# -*- coding: utf-8 -*-
"""
Created on Wed May 26 11:33:21 2021

@author: pietr
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as sc
from SPICY.spclass import mesh_lab
import pickle
import os
import shutil
GAMMA=10
rc=0.1
gamma=1.256431
rnd = np.random.default_rng(seed=39)
rndu = np.random.default_rng(seed=47)
rndv = np.random.default_rng(seed=42)
cTH=rc**2/gamma
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
pixel_2=1024*1024#Supposing a camera with 1024 pixel
ppp=np.array([0.002])
N=np.array(np.floor(ppp*pixel_2),dtype=int)
Noise=np.array([0,0.01])
dir = 'Result'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
dir = 'Result'+os.sep+'Vortex pickles'
os.mkdir(dir)
for k in np.arange(len(N)):
    number_particle=N[k]
    for NOISE in Noise:
            X=rnd.random(number_particle)-0.5
            Y=rnd.random(number_particle)-0.5
            TH=np.arctan2(Y,X)
            R=np.sqrt(X**2+Y**2)
            VTHREAL=GAMMA/(2*np.pi*R)*(1-np.exp(-R**2/cTH))
            u=VTHREAL*np.sin(TH)
            v=-VTHREAL*np.cos(TH)
            u=u+(2*rndu.random(u.shape)-1)*NOISE*u
            v=v+(2*rndv.random(v.shape)-1)*NOISE*v
            vortex=mesh_lab([u,v],[X,Y])
            NC=50
            XCON1=np.linspace(-0.5,0.5,NC)
            YCON1=-np.ones(NC)*0.5
            XCON11=np.array([XCON1[0]])
            YCON11=np.array([YCON1[0]])
            XCON12=XCON1[1::]
            YCON12=YCON1[1::]
            XCON2=np.linspace(-0.5,0.5,NC)
            YCON2=np.ones(NC)*0.5
            XCON3=-np.ones(NC)*0.5
            YCON3=np.linspace(-0.5,0.5,NC)
            XCON4=np.ones(NC)*0.5
            YCON4=np.linspace(-0.5,0.5,NC)
            XCON=[XCON11,XCON12,XCON2,XCON3,XCON4]
            YCON=[YCON11,YCON12,YCON2,YCON3,YCON4]
            con_u=['only_divergence','only_divergence','only_divergence','only_divergence','only_divergence']
            con_v=['only_divergence','only_divergence','only_divergence','only_divergence','only_divergence']
            CON=[con_u,con_v]
            vortex.velocities_constraint_definition([XCON,YCON,CON])
            vortex.clustering_velocities([6,10],cap=15,mincluster=[True,False],el=0.88)
            vortex.approximation_velocities(DIV=1,rcond=1e-12)
            rho=1
            mu=0
            TH1=np.arctan2(YCON11,XCON11)
            R1=np.sqrt(XCON11**2+YCON11**2)
            VTHREAL1=GAMMA/(2*np.pi*R1)*(1-np.exp(-R1**2/cTH))
            P11=-np.array([0.5*rho*VTHREAL1**2-rho*GAMMA**2/(4*np.pi**2*cTH)*(sc.exp1(R1**2/cTH)-sc.exp1(2*R1**2/cTH))])
            BCD=[P11,'Neumann','Neumann','Neumann','Neumann']
            n3=np.vstack((-np.ones(NC),np.zeros(NC)))
            n4=np.vstack((np.ones(NC),np.zeros(NC)))
            n11='None'
            n12=np.vstack((np.zeros(NC-1),-np.ones(NC-1)))
            n2=np.vstack((np.zeros(NC),np.ones(NC)))
            n=[n11,n12,n2,n3,n4]
            vortex.pressure_boundary_conditions(rho,mu,[XCON,YCON,BCD],n)
            vortex.clustering_pressure()
            vortex.pressure_computation(rcond=1e-12)
            #Remove old data and create new folder if not present
            with open(dir+os.sep+'vortex_data_NOISE='+str(NOISE)+'particlesperpixel'+str(ppp[k])+'.pkl', 'wb') as outp:
                 pickle.dump(vortex, outp, pickle.HIGHEST_PROTOCOL)