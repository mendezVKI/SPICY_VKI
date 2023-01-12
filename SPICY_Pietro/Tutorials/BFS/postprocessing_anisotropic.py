#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 19:35:42 2022

@author: sperotto
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import shutil
dir = 'Result'+os.sep+'BFSImg'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.close('all')
NOISE=0.05
with open('Result'+os.sep+'BFS pickles'+os.sep+'turbulent_anisotropic_NOISE='+str(NOISE)+'.pkl', 'rb') as pixk:
 turbulentBFS = pickle.load(pixk)
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

#Extrapolate data
Vcalc=turbulentBFS.extrapolate_velocities([X,Y,Z])
RScalc=turbulentBFS.extrapolate_RS([X,Y,Z])
Pcalc=turbulentBFS.extrapolate_pressure([X,Y,Z])
DIVcalc=turbulentBFS.extrapolate_divergence([X,Y,Z])

#Print the relative error
print('l_2 norm')
print('Error on u = '+str(np.linalg.norm(u-Vcalc[0])/np.linalg.norm(u)))
print('Error on v = '+str(np.linalg.norm(v-Vcalc[1])/np.linalg.norm(v)))
print('Error on w = '+str(np.linalg.norm(w-Vcalc[2])/np.linalg.norm(w)))
print('Error on divergence = '+str(np.linalg.norm(DIVcalc)/len(DIVcalc)))
print('Error on RSXX = '+str(np.linalg.norm(RSXX-RScalc[0])/np.linalg.norm(RSXX)))
print('Error on RSYY = '+str(np.linalg.norm(RSYY-RScalc[1])/np.linalg.norm(RSYY)))
print('Error on RSZZ = '+str(np.linalg.norm(RSZZ-RScalc[2])/np.linalg.norm(RSZZ)))
print('Error on RSXY = '+str(np.linalg.norm(RSXY-RScalc[3])/np.linalg.norm(RSXY)))
print('Error on RSXZ = '+str(np.linalg.norm(RSXZ-RScalc[4])/np.linalg.norm(RSXZ)))
print('Error on RSYZ = '+str(np.linalg.norm(RSYZ-RScalc[5])/np.linalg.norm(RSYZ)))
print('Error on p = '+str(np.linalg.norm(p-Pcalc)/np.linalg.norm(p)))

#Plot absolute error
print('L_2 norm')
print('Error on u = '+str(np.linalg.norm(u-Vcalc[0])))
print('Error on v = '+str(np.linalg.norm(v-Vcalc[1])))
print('Error on w = '+str(np.linalg.norm(w-Vcalc[2])))
print('Error on divergence = '+str(np.linalg.norm(DIVcalc)))
print('Error on RSXX = '+str(np.linalg.norm(RSXX-RScalc[0])))
print('Error on RSYY = '+str(np.linalg.norm(RSYY-RScalc[1])))
print('Error on RSZZ = '+str(np.linalg.norm(RSZZ-RScalc[2])))
print('Error on RSXY = '+str(np.linalg.norm(RSXY-RScalc[3])))
print('Error on RSXZ = '+str(np.linalg.norm(RSXZ-RScalc[4])))
print('Error on RSYZ = '+str(np.linalg.norm(RSYZ-RScalc[5])))
print('Error on p = '+str(np.linalg.norm(p-Pcalc)))
z0=-np.amin(np.abs(Z))

#Plot quiver
plt.rc('xtick',labelsize=10)
plt.rc('ytick',labelsize=10)
plt.figure(1,figsize=(5,2))
plt.quiver(X[Z==z0]*10,Y[Z==z0]*10,u[Z==z0],v[Z==z0])
plt.ylim([0,2])
plt.xlim([0,14])
plt.xlabel(r'$x[cm]$',fontsize=10)
plt.ylabel(r'$y[cm]$',fontsize=10)
plt.tight_layout()
plt.savefig(dir+os.sep+'quiver.pdf',dpi=500)
plt.show()
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
XX,YY=np.meshgrid(np.linspace(0,14,100),np.linspace(0,2,100))
Vplot=turbulentBFS.extrapolate_velocities([XX/10,YY/10,z0*np.ones(np.shape(XX))])
Pplot=turbulentBFS.extrapolate_pressure([XX/10,YY/10,z0*np.ones(np.shape(XX))])

#Plot velocity u
plt.figure(2,figsize=(5,2))
plt.pcolormesh(XX,YY,Vplot[0],vmin=-0.30270901092556035,vmax=0.629972756)
plt.colorbar()
plt.title(r'Velocity $\tilde{\overline{u}}[m/s]$',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'u_RBF.pdf',dpi=500)
plt.show()

#Plot velocity v
plt.figure(3,figsize=(5,2))
plt.pcolormesh(XX,YY,Vplot[1],vmin=-0.050448777854829964,vmax=0.07723723750439818)
plt.colorbar()
plt.title(r'Velocity $\tilde{\overline{v}}[m/s]$',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'v_RBF.pdf',dpi=500)
plt.show()

#Plot velocity w
plt.figure(4,figsize=(5,2))
plt.pcolormesh(XX,YY,Vplot[2],vmin=-0.013855134499068803,vmax=0.013206313830300607)
plt.colorbar()
plt.title(r'Velocity $\tilde{\overline{w}}[m/s]$',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'w_RBF.pdf',dpi=500)
plt.show()

#Plot pressure
plt.figure(5,figsize=(5,2))
plt.pcolormesh(XX,YY,Pplot,vmin=-0.25299520084016597,vmax=-0.0526839271)
plt.colorbar()
plt.title(r'Pressure $\tilde{\overline{p}}[Pa]$',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'p_RBF.pdf',dpi=500)
plt.show()

#Plot in plane components
plt.figure(6,figsize=(5,2))
plt.pcolormesh(XX,YY,np.sqrt(Vplot[0]**2+Vplot[1]**2),vmax=0.6313489195812758,vmin=0)
plt.colorbar()
plt.title(r'In plane velocity components $[m/s]$',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.quiver(XX[::4,::4],YY[::4,::4],Vplot[0][::4,::4],Vplot[1][::4,::4],color='white')
plt.savefig(dir+os.sep+'uv_RBF.pdf',dpi=500)
plt.show()

#Velocity profiles 
A=np.loadtxt('Anistotropic data/stats_averages/stats_averages_z_1_8.text')
x=A[:,0]
y=A[:,1]
u=A[:,3]
v=A[:,4]
w=A[:,5]
logic=np.logical_and(y<=0,y>=-2)
x=x[logic]
y=y[logic]
u=u[logic]
v=v[logic]
x1=x[x==1]
y1=y[x==1]+2
u1=u[x==1]
x2=x[x==6]
y2=y[x==6]+2
u2=u[x==6]
x3=x[x==13]
y3=y[x==13]+2
u3=u[x==13]
V1=turbulentBFS.extrapolate_velocities([x1/10,y1/10,z0*np.ones(len(y1))])
plt.figure(7)
plt.plot(u1,y1,'r--')
plt.plot(u1*0.95,y1,'b-')
plt.plot(u1*1.05,y1,'b-')
plt.plot(V1[0],y1,'k-')
plt.ylim([0,2])
plt.savefig(dir+os.sep+'profile1.pdf',dpi=300)
plt.show()
V2=turbulentBFS.extrapolate_velocities([x2/10,y2/10,z0*np.ones(len(y2))])
plt.figure(8)
plt.plot(u2,y2,'r--')
plt.plot(u2*0.95,y2,'b-')
plt.plot(u2*1.05,y2,'b-')
plt.plot(V2[0],y2,'k-')
plt.ylim([0,2])
plt.savefig(dir+os.sep+'profile2.pdf',dpi=300)
plt.show()
V3=turbulentBFS.extrapolate_velocities([x3/10,y3/10,z0*np.ones(len(y3))])
plt.figure(9)
plt.plot(u3,y3,'r--')
plt.plot(u3*0.95,y3,'b-')
plt.plot(u3*1.05,y3,'b-')
plt.plot(V3[0],y3,'k-')
plt.ylim([0,2])
plt.savefig(dir+os.sep+'profile3.pdf',dpi=300)
plt.show()
XX,YY=np.meshgrid(np.linspace(0,14,100),np.linspace(0,2,100))
RScalc=turbulentBFS.extrapolate_RS([XX/10,YY/10,z0*np.ones(np.shape(XX))])
TKEcalc=0.5*(RScalc[0]+RScalc[1]+RScalc[2])
plt.figure(10,figsize=(5,2))
plt.pcolormesh(XX,YY,TKEcalc,vmin=0,vmax=0.07154504113747849)
plt.colorbar()
plt.title(r'$\tilde{k}[m^2/s^2]$',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'TKE_RBF.pdf',dpi=500)
plt.show()
A=np.loadtxt('Anistotropic data/stats_averages/stats_averages_z_1_8.text')
C=np.loadtxt('Anistotropic data/stats_rms/stats_rms_z_1_8.text')
x=A[:,0]
y=A[:,1]
u=A[:,3]
v=A[:,4]
w=A[:,5]
p=A[:,6]
XX,YY=np.meshgrid(np.linspace(0,14,100),np.linspace(-2,0,100))
from scipy.interpolate import griddata as gd
pplot=gd((x,y), p, (XX,YY), method='cubic')
print('error out of sample Pressure'+str(np.linalg.norm(pplot-Pplot)/np.linalg.norm(pplot)))
plt.close('all')
