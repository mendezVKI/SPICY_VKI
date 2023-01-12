# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:05:35 2021

@author: pietr
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
import os
import scipy.io
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
dir = 'Result'+os.sep+'CylinderImg'
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
Errv=np.zeros(20)
Errp=np.zeros(20)
Vmagn=np.sqrt(vx**2+vy**2)
i=0
for NOISE in np.round(np.linspace(0,0.1,2),2):#loop the noise
 for N in (4.0,6.0):#loop over the two gaussian collocation
  for nparticle in np.int16(np.linspace(6000,18755,5)):#loop the number of particles
 
      with open('Result'+os.sep+'Cylinder pickles'+os.sep+'cyl_data_NOISE='+str(NOISE)+'numberofparticles'+str(nparticle)+'number'+str(N)+'.pkl','rb') as pixk:
       cyl = pickle.load(pixk)
      U=cyl.extrapolate_velocities([X,Y])
      Errv[i]=np.linalg.norm(Vmagn-np.sqrt(U[0]**2+U[1]**2))/np.linalg.norm(Vmagn)
      Pcalc=cyl.extrapolate_pressure([X,Y])
      Errp[i]=np.linalg.norm(P-Pcalc)/np.linalg.norm(Pcalc)
      i=i+1
Errv1=Errv[:5:]
Errv2=Errv[5:10:]
Errv3=Errv[10:15:]
Errv4=Errv[15::]
Errp1=Errp[:5:]
Errp2=Errp[5:10:]
Errp3=Errp[10:15:]
Errp4=Errp[15::]
plt.figure()
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errv1,color='black',  marker='.',
     markerfacecolor='red', markersize=5)
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errv2,color='black',  marker='^',
     markerfacecolor='green', markersize=5)
plt.legend(['N=[4,10,20]','N=[6,10,20]'])
plt.title(r'$L_2$ relative norm of error velocity for different seeding q=0.0')
plt.savefig(dir+os.sep+'Plotarticle1.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
plt.figure()
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errv3,color='black',  marker='.',
     markerfacecolor='red', markersize=5)
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errv4,color='black',  marker='^',
     markerfacecolor='green', markersize=5)
plt.legend(['N=[4,10,20]','N=[6,10,20]'])
plt.title(r'$L_2$ relative norm of error velocity for different seeding q=0.1')
plt.savefig(dir+os.sep+'Plotarticle2.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
plt.figure()
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errp1,color='black',  marker='.',
     markerfacecolor='red', markersize=5)
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errp2,color='black',  marker='^',
     markerfacecolor='green', markersize=5)
plt.legend(['N=[4,10,20]','N=[6,10,20]'])
plt.title(r'$L_2$ relative norm of error pressure for different seeding q=0.0')
plt.savefig(dir+os.sep+'Plotarticle3.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
plt.figure()
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errp3,color='black',  marker='.',
     markerfacecolor='red', markersize=5)
plt.plot(np.int16(np.linspace(6000,18755,5))/(1024*2180),Errp4,color='black',  marker='^',
     markerfacecolor='green', markersize=5)
plt.legend(['N=[4,10,20]','N=[6,10,20]'])
plt.title(r'$L_2$ relative norm of error pressure for different seeding q=0.1')
plt.savefig(dir+os.sep+'Plotarticle4.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
