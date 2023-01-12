# -*- coding: utf-8 -*-
"""
Created on Sun Aug  8 17:41:16 2021

@author: pietr
"""
import numpy as np
#import os
import matplotlib.pyplot as plt 
import scipy.io
import pickle
from SPICY.spclass import mesh_lab
import os
import shutil
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
tipo='float64' #put this to change the kind of data used
dir = 'Result'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
dir = 'Result'+os.sep+'Cylinder pickles'
os.mkdir(dir)
#%% 0. Prepare the data from the FluenSol.mat
# This test case is elaborated from:
# https://github.com/Raocp/PINN-laminar-flow
    
data=scipy.io.loadmat('FluentSol.mat')#extract datas from CFD ansys
X = data['x'];  Y = data['y']; # Mesh Points
P = data['p']; vx = data['vx']; vy = data['vy'] # Variables
#%% Data for the simulation
R=0.05#Radius circle
H=0.41#Height ROI
L=1.1#Length ROI
# Reshape results in vectors
X=X.reshape(-1)#x coordinate of datas
Y=Y.reshape(-1)#y coordinate of datas
P=P.reshape(-1)#P  datas
vx=vx.reshape(-1)#vx datas
vy=vy.reshape(-1)#vy datas
X=X.astype(tipo)
Y=Y.astype(tipo)
P=P.astype(tipo)
vx=vx.astype(tipo)
vy=vy.astype(tipo)
#logical to remove the point which are on the boundaries
WALLBOOL=np.logical_and(vx==0,vy==0)
WALLBOOL=np.logical_or(WALLBOOL,X==0)
circlebool=np.logical_and(np.logical_and(vx==0,vy==0),np.logical_and(Y!=0,Y!=H))
#Extract the final dataset
Xg=X[np.invert(WALLBOOL)]
Yg=Y[np.invert(WALLBOOL)]
Pg=P[np.invert(WALLBOOL)]
vxg=vx[np.invert(WALLBOOL)]
vyg=vy[np.invert(WALLBOOL)]
sample_arr = [True, False]
# Create a numpy array with random True or False of size 10
for NOISE in np.round(np.linspace(0,0.1,2),2):
 for N1 in (4.0,6.0):#loop over the two gaussian collocation
  for nparticle in np.int16(np.linspace(6000,18755,5)):
    Xg=X[np.invert(WALLBOOL)]
    Yg=Y[np.invert(WALLBOOL)]
    Pg=P[np.invert(WALLBOOL)]
    vxg=vx[np.invert(WALLBOOL)]
    vyg=vy[np.invert(WALLBOOL)]
    if nparticle!=18755:
        np.random.seed(47)
        bool_arr=np.random.randint(low=0,high=len(Xg),size=nparticle)
        Xg=Xg[bool_arr]
        Yg=Yg[bool_arr]
        Pg=Pg[bool_arr]
        vxg=vxg[bool_arr]
        vyg=vyg[bool_arr]
    nparticles=len(Xg)
    #%%Generate noise
    rng1 = np.random.default_rng(seed=47)
    rng2 = np.random.default_rng(seed=32)
    vxg=vxg+(2*rng1.random(vxg.shape)-1)*NOISE*vxg
    vyg=vyg+(2*rng2.random(vxg.shape)-1)*NOISE*vyg
    vxg=vxg.astype(tipo)
    vyg=vyg.astype(tipo)
    #%%Start the calculation
    cyl=mesh_lab([vxg,vyg],[Xg,Yg])#The class mesh_lab is formed as input we have the velocities and the relatives velocity
    NC=150 #number of collocation points for the every BC (constraints)
    # X and Y coordinates of upper boundary  
    YCON1=H*np.ones(NC,dtype=tipo)# Y coordinate of upper boundary
    XCON1=np.linspace(0,L,NC,dtype=tipo)# X coordinate of upper boundary
    # X and Y coordinates of Bottom Boundary  
    YCON3=np.zeros(NC-1,dtype=tipo)# Y coordinate of bottom boundary
    XCON3=np.linspace(L/NC,L,NC-1,dtype=tipo)# X coordinate of bottom boundary
    # X and Y coordinates of the Cylinder
    alphaT=np.linspace(0,2*np.pi*(NC-1)/NC,NC-1,dtype=tipo) # angular spacing
    alphaT=np.hstack((alphaT,np.pi,np.pi/2,np.pi*3/2))
    XCON5=0.2+R*np.cos(alphaT)
    YCON5=0.2+R*np.sin(alphaT)
    # X and Y coordinates of the Inlet
    XCON4=np.hstack((np.zeros(NC,dtype=tipo)))#Pay attention another hide constraint is present 
    #to ensure the constant value of the velocity before the inlet, therefore some constraint equal to the one
    #of the inlet is added before the entry, that just why there is still no Neumann constraint on velocities
    YCON4=np.hstack((np.linspace(0,H,NC,dtype=tipo)))
    #At the outlet there are only divergence free constraint
    XCON2=L*np.ones(NC-2,dtype=tipo)#points where contraint are applied (only divergence free)
    YCON2=np.linspace(0+H/NC,H-H/NC,NC-2,dtype=tipo)#points where contraint are applied (only divergence free)
    boolcyl=(Xg-0.2)**2+(Yg-0.2)**2<(4*R)**2
    XCON6=Xg[boolcyl]
    YCON6=Yg[boolcyl]
    # Stack all the Standard constraint points Points
    XCON=[XCON1,XCON2,XCON3,XCON4,XCON5]#points where contraint are applied
    YCON=[YCON1,YCON2,YCON3,YCON4,YCON5]#points where contraint are applied
    CONu1=np.zeros(len(XCON1),dtype=tipo)#conditions (no slip and inlet) for u
    CONv1=np.zeros(len(XCON1),dtype=tipo)#conditions (no slip and inlet) for v
    CONu3=np.zeros(len(XCON3),dtype=tipo)#conditions (no slip and inlet) for u
    CONv3=np.zeros(len(XCON3),dtype=tipo)#conditions (no slip and inlet) for v
    CONu4=4*(H-YCON4)*YCON4/H**2
    CONv4=np.zeros(len(XCON4),dtype=tipo)
    CONu5=np.zeros(len(XCON5),dtype=tipo)#conditions (no slip and inlet) for u
    CONv5=np.zeros(len(XCON5),dtype=tipo)#conditions (no slip and inlet) for v
    CONU=[CONu1,'only_div',CONu3,CONu4,CONu5]#Definition of constraint of velocity list
    CONV=[CONv1,'only_div',CONv3,CONv4,CONv5]
    CON=[CONU,CONV]#This is the input accepted by the code
    cyl.velocities_constraint_definition([XCON,YCON,CON])#Define the constraint in a way that is usable by the subroutine of mesh_lab
    N=[N1,10,20]
    #with the change in the definition of the cap we have to change it for different N
    if N1==4:
        C=58.56199661501407
    elif N1==6:
        C=47.07690424526196
    cyl.clustering_velocities(N,C,[True,True,True,False],0.83)#Creates the clustering of the point
    cyl.approximation_velocities(rcond=1e-13,DIV=1)#Approximate the velocity field
    XBC1=np.linspace(0.0,L,NC,dtype=tipo)#X position of BC point in the lower edge
    YBC1=np.zeros(NC,dtype=tipo)#Y position of BC point in the lower edge
    XBC2=L*np.ones(NC,dtype=tipo)#X position of BC point in the right edge
    YBC2=np.linspace(0,H,NC,dtype=tipo)#Y position of BC point in the right edge
    XBC3=np.linspace(0.0,L,NC,dtype=tipo)#X position of BC point in the upper edge
    YBC3=H*np.ones(NC,dtype=tipo)#Y position of BC point in the upper edge
    XBC4=0.0*np.ones(NC,dtype=tipo)#X position of BC point in the left edge
    YBC4=np.linspace(0,H,NC,dtype=tipo)#X position of BC cons point in the left edge
    alphaP=np.linspace(0,2*np.pi*(NC-1)/NC,NC,dtype=tipo) 
    alphaP=np.hstack((alphaP,np.pi,np.pi/2,np.pi*3/2))
 
    XBC5=0.2+R*np.cos(alphaP)#X position of BC point in the cylinder
    YBC5=0.2+R*np.sin(alphaP)#Y position of BC point in the cylinder
    XBC=[XBC1,XBC2,XBC3,XBC4,XBC5]#Stacking all the constrained points
    YBC=[YBC1,YBC2,YBC3,YBC4,YBC5]
    #define the various type of BC
    NEU='Neumann'
    NEUW='Neumann Wall'
    BCD=[NEUW,np.zeros(len(XBC2)),NEUW,NEU,NEUW]#different type of BC stacked 
    #Array for normal direction to the walls
    n1=np.vstack((np.zeros(len(XBC1)),-np.ones(len(XBC1))))
    n2=np.vstack((np.ones(len(XBC2)),np.zeros(len(XBC2))))
    n3=np.vstack((np.zeros(len(XBC3)),np.ones(len(XBC3))))
    n4=np.vstack((-np.ones(len(XBC2)),np.zeros(len(XBC2))))
    n5=np.vstack((-np.cos(alphaP),-np.sin(alphaP)))
    n=[n1,n2,n3,n4,n5]
    #Physics values
    rho=1
    mu=2*1e-2
    cyl.pressure_boundary_conditions(rho,mu,[XBC,YBC,BCD],n)#pressure boundary conditions
    cyl.clustering_pressure()#pressure clustering (right now the code just reuse the collocation point of the velocity by changing BC point if needed)
    cyl.pressure_computation(rcond=1e-13)#compute pressure
    ##Create the logical to individuate the point in the slices
    with open(dir+os.sep+'cyl_data_NOISE='+str(NOISE)+'numberofparticles'+str(nparticles)+'number'+str(np.float64(N[0]))+'.pkl','wb') as outp:
             pickle.dump(cyl, outp, pickle.HIGHEST_PROTOCOL)
