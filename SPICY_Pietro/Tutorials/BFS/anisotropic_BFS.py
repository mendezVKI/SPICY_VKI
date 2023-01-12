# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:01:40 2021

@author: pietr
"""
import numpy as np
import matplotlib.pyplot as plt
from SPICY.spclass import mesh_lab
import pickle
import os
import shutil
plt.close('all')

#Extracting the data
A=np.loadtxt('Anistotropic data/database.txt')
X=A[0,:]
Y=A[1,:]
Z=A[2,:]
u=A[3,:]
v=A[4,:]
w=A[5,:]
p=A[6,:]
RSXX=A[7,:]
RSXY=A[8,:]
RSXZ=A[9,:]
RSYY=A[10,:]
RSYZ=A[11,:]
RSZZ=A[12,:]

#Defining the physical properties
mu=1/32000
rho=1

#Adding noise
rngu = np.random.default_rng(seed=47)
rngv = np.random.default_rng(seed=29)
rngw = np.random.default_rng(seed=35)
rngRS = np.random.default_rng(seed=35)
NOISE=0.05
u=u+(2*rngu.random(u.shape)-1)*NOISE*u
v=v+(2*rngv.random(v.shape)-1)*NOISE*v
w=w+(2*rngw.random(w.shape)-1)*NOISE*w
RSXX=RSXX+(2*rngRS.random(w.shape)-1)*NOISE*RSXX
RSXY=RSXY+(2*rngRS.random(w.shape)-1)*NOISE*RSXY
RSXZ=RSXZ+(2*rngRS.random(w.shape)-1)*NOISE*RSXZ
RSYY=RSYY+(2*rngRS.random(w.shape)-1)*NOISE*RSYY
RSYZ=RSYZ+(2*rngRS.random(w.shape)-1)*NOISE*RSYZ
RSZZ=RSZZ+(2*rngRS.random(w.shape)-1)*NOISE*RSZZ

#Inizialing meshlab
turbulentBFS=mesh_lab([u,v,w],[X,Y,Z],ST=[RSXX,RSYY,RSZZ,RSXY,RSXZ,RSYZ],model='RANSA')

#Defining constraints and constraint points
NC=120
NC2=0
con_u=[np.zeros(NC+NC2),np.zeros(NC+NC2),'only_div','only_div','only_div','only_div']
con_v=[np.zeros(NC+NC2),np.zeros(NC+NC2),'only_div','only_div','only_div','only_div']
con_w=[np.zeros(NC+NC2),np.zeros(NC+NC2),'only_div','only_div','only_div','only_div']
XCON1=np.zeros(NC)
YCON1=rngu.random(NC)*(np.amax(Y))
ZCON1=rngv.random(NC)*(np.amax(Z)-np.amin(Z))+np.amin(Z)
YCON2=np.zeros(NC)
XCON2=rngu.random(NC)*np.amax(X)
ZCON2=rngv.random(NC)*(np.amax(Z)-np.amin(Z))+np.amin(Z)
XCON3=np.ones(NC)*np.amax(X)
YCON3=rngu.random(NC)*np.amax(Y)
ZCON3=rngv.random(NC)*(np.amax(Z)-np.amin(Z))+np.amin(Z)
YCON4=np.ones(NC)*np.amax(Y)
XCON4=rngu.random(NC)*np.amax(X)
ZCON4=rngv.random(NC)*(np.amax(Z)-np.amin(Z))+np.amin(Z)
XCON1=np.hstack((XCON1,np.zeros(NC2)))
YCON1=np.hstack((YCON1,np.linspace(np.amin(Y),np.amax(Y),NC2)))
ZCON1=np.hstack((ZCON1,np.ones(NC2)*Z[np.amax(np.abs(Z))==np.abs(Z)][0]))
XCON2=np.hstack((XCON2,np.linspace(np.amin(X),np.amax(X),NC2)))
YCON2=np.hstack((YCON2,np.zeros(NC2)))
ZCON2=np.hstack((ZCON2,np.ones(NC2)*Z[np.amax(np.abs(Z))==np.abs(Z)][0]))
XCON3=np.hstack((XCON3,np.ones(NC2)*np.amax(X)))
YCON3=np.hstack((YCON3,np.linspace(np.amin(Y),np.amax(Y),NC2)))
ZCON3=np.hstack((ZCON3,np.ones(NC2)*Z[np.amax(np.abs(Z))==np.abs(Z)][0]))
XCON4=np.hstack((XCON4,np.linspace(np.amin(X),np.amax(X),NC2)))
YCON4=np.hstack((YCON4,np.zeros(NC2)))
ZCON4=np.hstack((ZCON4,np.ones(NC2)*Z[np.amax(np.abs(Z))==np.abs(Z)][0]))
ZCON5=np.ones(NC)*np.amin(Z)
XCON5=rngu.random(NC)*np.amax(X)
YCON5=rngv.random(NC)*np.amax(Y)
ZCON6=np.ones(NC)*np.amax(Z)
XCON6=rngu.random(NC)*np.amax(X)
YCON6=rngv.random(NC)*np.amax(Y)
CON=[con_u,con_v,con_w]
XCON=[XCON1,XCON2,XCON3,XCON4,XCON5,XCON6]
YCON=[YCON1,YCON2,YCON3,YCON4,YCON5,YCON6]
ZCON=[ZCON1,ZCON2,ZCON3,ZCON4,ZCON5,ZCON6]

#Giving the constraint for Reynolds stresses and velocities
turbulentBFS.velocities_constraint_definition([XCON,YCON,ZCON,CON])
turbulentBFS.RS_constraint_definition([[XCON1,XCON2],[YCON1,YCON2],[ZCON1,ZCON2],[np.zeros(NC+NC2),np.zeros(NC+NC2)]])

#Clustering
turbulentBFS.clustering_velocities([1,3],cap=15, mincluster=[True,True,True,True],el=0.78)

#Fitting the velocities
turbulentBFS.approximation_velocities(DIV=1e-2,rcond=1e-13,method='fullcho')

#Defining the normal direction
n1=np.vstack(( -np.ones(NC+NC2), np.zeros(NC+NC2), np.zeros(NC+NC2)))
n6=np.vstack((np.zeros(NC),np.zeros(NC),+np.ones(NC)))
n5=np.vstack((np.zeros(NC),np.zeros(NC),-np.ones(NC)))
n2=np.vstack((np.zeros(NC+NC2),-np.ones(NC+NC2),np.zeros(NC+NC2)))
n4=np.vstack((np.zeros(NC+NC2),np.ones(NC+NC2),np.zeros(NC+NC2)))
n3=np.vstack((np.ones(NC+NC2),np.zeros(NC+NC2),np.zeros(NC+NC2)))

#Prepare the interpolation for the Dirichlet boundary condition
from scipy.interpolate import griddata as gd
ND=10
ZCOND,XCOND=np.meshgrid(Z[np.abs(Z)==np.amin(np.abs(Z))][0],
                        np.linspace(0,np.amax(X),ND))
YCOND=np.amin(Y)*1.05*np.ones(np.shape(XCOND))
n=[n1,n2,n3,n4,n5,n6,['NA']]
DIRICHLET=gd((X,Y,Z), p, (XCOND,YCOND,ZCOND), method='linear')
XCOND=XCOND.reshape(-1)
YCOND=YCOND.reshape(-1)
ZCOND=ZCOND.reshape(-1)
DIRICHLET=DIRICHLET.reshape(-1)
logic=np.invert(np.isnan(DIRICHLET))
XCOND=XCOND[logic]
YCOND=YCOND[logic]
ZCOND=ZCOND[logic]

#Preparing the Boudary conditions
DIRICHLET=DIRICHLET[logic]
DIRICHLET=DIRICHLET+(2*rngRS.random(DIRICHLET.shape)-1)*NOISE*DIRICHLET
XCON=[XCON1,XCON2,XCON3,XCON4,XCON5,XCON6,XCOND]
YCON=[YCON1,YCON2,YCON3,YCON4,YCON5,YCON6,YCOND]
ZCON=[ZCON1,ZCON2,ZCON3,ZCON4,ZCON5,ZCON6,ZCOND]
BCD=['Neumann Wall','Neumann Wall','Neumann','Neumann',
     'Neumann','Neumann',DIRICHLET]

#Creating the boundary condition in mesh lab
turbulentBFS.pressure_boundary_conditions(rho,mu,[XCON,YCON,ZCON,BCD],n)

#Clustering for the pressure (if the condition are on the same point as in 
#the velocity this just re assigned the previous collocation points)
turbulentBFS.clustering_pressure()

#Compute the pressure 
turbulentBFS.pressure_computation(rcond=1e-13,method='fullcho')
dir = 'Result'+os.sep+'BFS pickles'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
#Save the picle of the class
with open(dir+os.sep+'turbulent_anisotropic_NOISE='+str(NOISE)+'.pkl', 'wb') as outp:
         pickle.dump(turbulentBFS, outp, pickle.HIGHEST_PROTOCOL)