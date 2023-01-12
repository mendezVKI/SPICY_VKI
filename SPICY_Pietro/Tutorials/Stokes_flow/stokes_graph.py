#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 19:12:17 2021

@author: sperotto
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import SPICY.Matrix as Matrix
import os
import shutil
#Plot setting
plt.close('all')
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
dir = 'Result'+os.sep+'StokesImg'
if os.path.isdir(dir):
else:
  os.mkdir(dir)
#Physical dimension
D=1
R=D/2
mu=1
U_0=1

#This two must be changed depending on the result one want to plot
NOISE=0.05
nparticles=40000

#read the pickle
with open('Result'+os.sep+'Stokes pickles'+os.sep+'stokes_data_NOISE='+str(NOISE)+'numberofparticles'+str(nparticles)+'.pkl', 'rb') as pixk:
 stokes = pickle.load(pixk)
 
#Create the correct velocity field from equation
r=np.sqrt(stokes.XG**2+stokes.YG**2+stokes.ZG**2)
phi=np.arctan2(stokes.YG,stokes.XG)
th=np.arctan2(np.sqrt(stokes.XG**2+stokes.YG**2),(stokes.ZG))
N=len(stokes.XG)
v_r=U_0*(1-3/2*R/r+0.5*(R/r)**3)*np.cos(th) #theorical velocities
v_th=-U_0*(1-3/4*R/r-0.25*(R/r)**3)*np.sin(th)
ureal=v_r*np.cos(phi)*np.sin(th)+v_th*np.cos(phi)*np.cos(th)
vreal=v_r*np.sin(phi)*np.sin(th)+v_th*np.cos(th)*np.sin(phi)
wreal=v_r*np.cos(th)-v_th*np.sin(th)
Pcorr=-3/2*(mu*U_0*R/r**2)*np.cos(th)
Ureal=np.sqrt(ureal**2+vreal**2+wreal**2)

#Extrapolate velocities
u,v,w=stokes.extrapolate_velocities([stokes.XG,stokes.YG,stokes.ZG])
U=np.sqrt(u**2+v**2+w**2)

#Extrapolate pressure
P=stokes.extrapolate_pressure([stokes.XG,stokes.YG,stokes.ZG])

#Compute the errors on in sample points
print('Error un velocity u: '+str(np.linalg.norm(ureal-u)/np.linalg.norm(ureal)))
print('Error un velocity v: '+str(np.linalg.norm(vreal-v)/np.linalg.norm(vreal)))
print('Error un velocity w: '+str(np.linalg.norm(wreal-w)/np.linalg.norm(wreal)))
print('Error un velocity P: '+str(np.linalg.norm(Pcorr-P)/np.linalg.norm(Pcorr)))
print('Error un velocity magnitude: '+str(np.linalg.norm(Ureal-U)/np.linalg.norm(Ureal)))

#Definition of a grif for the plot
Y,Z=np.meshgrid(np.linspace(-1,1,200),np.linspace(-1,1,200))

#Delete what is not part of the domain
logcyl=Z**2+Y**2<=(R*2)**2
Y=Y[logcyl]
Z=Z[logcyl]
logcyl=Z**2+Y**2>(R)**2
Y=Y[logcyl]
Z=Z[logcyl]
Y=Y.reshape(-1)
Z=Z.reshape(-1)

#Creat the correct values in the plot section
X=np.zeros(len(Y))
rplot=np.sqrt(X**2+Y**2+Z**2)
phiplot=np.arctan2(Y,X)
thplot=np.arctan2(np.sqrt(X**2+Y**2),(Z))
v_rplot=U_0*(1-3/2*R/rplot+0.5*(R/rplot)**3)*np.cos(thplot) #theorical velocities
v_thplot=-U_0*(1-3/4*R/rplot-0.25*(R/rplot)**3)*np.sin(thplot)
urealplot=v_rplot*np.cos(phiplot)*np.sin(thplot)+v_thplot*np.cos(phiplot)*np.cos(thplot)
vrealplot=v_rplot*np.sin(phiplot)*np.sin(thplot)+v_thplot*np.cos(thplot)*np.sin(phiplot)
wrealplot=v_rplot*np.cos(thplot)-v_thplot*np.sin(thplot)
Pcorrplot=-3/2*(mu*U_0*R/rplot**2)*np.cos(thplot)
P=stokes.extrapolate_pressure([X,Y,Z])

#Plot the results
uplot,vplot,wplot=stokes.extrapolate_velocities([X,Y,Z])
usp = np.linspace(0, 2 * np.pi, 100)
vsp = np.linspace(0, np.pi, 100)
xsp = 0.5 * np.outer(np.cos(usp), np.sin(vsp))
ysp = 0.5 * np.outer(np.sin(usp), np.sin(vsp))
zsp = 0.5* np.outer(np.ones(np.size(usp)), np.cos(vsp))
fig, ax = plt.subplots()
PS=plt.scatter(Y, Z,  s=10,c=np.abs(urealplot-uplot)+np.abs(vrealplot-vplot)+np.abs(wrealplot-wplot), alpha=0.9)
plt.plot(ysp,zsp,'ko')
ax.set_yticks([-1,-0.5,0,0.5,1])
ax.set_xticks([-1,-0.5,0,0.5,1])
ax.set_aspect('equal')

#Velocity error
eq1 = (r"$\left\|\hat{\mathbf{U}}(\mathbf{x})-\mathbf{\Phi_U}(\mathbf{x})\mathbf{w_U}\right\|_1$")
#plt.title(eq1, fontsize=18, y=1.05)
cb = plt.colorbar(PS,ax=[ax],location='right')
plt.ylabel(r'$\hat{z} [-]$',fontsize=18)
plt.xlabel(r'$\hat{y} [-]$',fontsize=18)
plt.savefig(dir+os.sep+'Errorstokesvelocity'+str(NOISE)+'.pdf',dpi=300,transparent=True,bbox_inches='tight')
fig, ax = plt.subplots()
UM=plt.scatter(Y, Z,  s=10, c=np.abs(Pcorrplot-P), alpha=0.9)
cb = plt.colorbar(UM,ax=[ax],location='right')
plt.plot(ysp,zsp,'ko')
ax.set_yticks([-1,-0.5,0,0.5,1])
ax.set_xticks([-1,-0.5,0,0.5,1])
ax.set_aspect('equal')

#Pressure error
eq2 = (r"$\left|\hat{p}(\mathbf{x})-\mathbf{\Phi}(\mathbf{x})\mathbf{w_P}\right|$")
#plt.title(eq2, fontsize=18)
plt.ylabel(r'$\hat{z} [-]$',fontsize=18)
plt.xlabel(r'$\hat{y} [-]$',fontsize=18)
plt.savefig(dir+os.sep+'Errorstokespressure'+str(NOISE)+'.pdf',dpi=300,transparent=True,bbox_inches='tight')
div=stokes.extrapolate_divergence([X,Y,Z])
plt.ylabel(r'$\hat{z} [-]$',fontsize=18)
plt.xlabel(r'$\hat{y} [-]$',fontsize=18)
fig, ax = plt.subplots()
PS=plt.scatter(Y, Z,  s=10,c=np.abs(div), alpha=0.9)
plt.plot(ysp,zsp,'ko')
ax.set_yticks([-1,-0.5,0,0.5,1])
ax.set_xticks([-1,-0.5,0,0.5,1])
ax.set_aspect('equal')

#Plot the divergence
eq3 = (r"$\left|\mathbf{D_{\nabla}}(\mathbf{x})\mathbf{w_U}\right|$")
#plt.title(eq3, fontsize=18)
plt.ylabel(r'$\hat{z} [-]$',fontsize=18)
plt.xlabel(r'$\hat{y} [-]$',fontsize=18)
cb = plt.colorbar(PS,ax=[ax],location='right')
plt.savefig(dir+os.sep+'Errorstokesdivergence'+str(NOISE)+'.pdf',dpi=300,transparent=True,bbox_inches='tight')
alphaplot2=np.linspace(0,2*np.pi,1000,endpoint=False)
Yplot2=R*np.sin(alphaplot2)
Zplot2=R*np.cos(alphaplot2)
Xplot2=np.zeros(len(Zplot2))
Psphere=stokes.extrapolate_pressure([Xplot2,Yplot2,Zplot2])
Pcorrplot=-3/2*(mu*U_0/R)*np.cos(alphaplot2)

#Plot the pressure cut
plt.figure()
plt.plot(alphaplot2*180/np.pi,Pcorrplot,'b-.')
plt.plot(alphaplot2*180/np.pi,Psphere,'k-')
plt.legend(['Reference', 'Computed'],fontsize=12)
plt.xlabel(r'$\theta [^o]$',fontsize=18)
plt.ylabel(r'$\hat{p} [-]$',fontsize=18)
plt.savefig(dir+os.sep+'Pcirclestokes'+str(NOISE)+'.pdf', dpi=300,transparent=True,bbox_inches='tight') 
plt.show()
dPdr=3*(mu*U_0/R**2)*np.cos(alphaplot2)
dPdX=Matrix.Der_RBF_Z3D(stokes.X_C_P,stokes.Y_C_P,stokes.Z_C_P,Xplot2,Yplot2,Zplot2,stokes.c_P,1e-13)
dPdY=Matrix.Der_RBF_Y3D(stokes.X_C_P,stokes.Y_C_P,stokes.Z_C_P,Xplot2,Yplot2,Zplot2,stokes.c_P,1e-13)
LAP=Matrix.LAP_RBF3D(stokes.X_C_vel,stokes.Y_C_vel,stokes.Z_C_vel,Xplot2,Yplot2,Zplot2,stokes.cvel,1e-13)
dPdrvel=mu*((LAP.dot(stokes.w_w))*np.cos(alphaplot2)+(LAP.dot(stokes.w_v))*np.sin(alphaplot2))
dPdrcomp=(dPdX.dot(stokes.w_P))*np.cos(alphaplot2)+(dPdY.dot(stokes.w_P))*np.sin(alphaplot2)

#Plot pressure normal derivatives
plt.figure()
plt.plot(alphaplot2*180/np.pi,dPdr,'b-.')
plt.plot(alphaplot2*180/np.pi,dPdrcomp,'k-')
plt.plot(alphaplot2*180/np.pi,dPdrvel,'r--')
plt.legend(['Reference', 'Computed','Projection of NS equations'],fontsize=12)
plt.xlabel(r'$\theta [^o]$',fontsize=18)
plt.ylabel(r'$\partial_{\hat{r}}\hat{p} [-]$',fontsize=18)
plt.savefig(dir+os.sep+'dPdrcirclestokes'+str(NOISE)+'.pdf', dpi=300,transparent=True,bbox_inches='tight') 
plt.show()
r=np.sqrt(stokes.XG**2+stokes.YG**2+stokes.ZG**2)
phi=np.arctan2(stokes.YG,stokes.XG)
th=np.arctan2(np.sqrt(stokes.XG**2+stokes.YG**2),(stokes.ZG))
N=len(stokes.XG)
r=np.linspace(0.5,1,1000)
th=np.pi/2
v_r=U_0*(1-3/2*R/r+0.5*(R/r)**3)*np.cos(th) #theorical velocities
v_th=-U_0*(1-3/4*R/r-0.25*(R/r)**3)*np.sin(th)
wreal=v_r*np.cos(th)-v_th*np.sin(th)
Z1=r*np.cos(th)
Y1=r*np.sin(th)
X1=np.zeros(1000)
_,_,w1=stokes.extrapolate_velocities([X1,Y1,Z1])

#plot the velocity reconstruction
plt.figure()
plt.plot(r,wreal,'b-.')
plt.plot(r,w1,'k-')
plt.legend(['Reference','computed'],fontsize=12)
plt.ylabel(r'$\hat{w} [-]$',fontsize=18)
plt.xlabel(r'$\hat{r} [-]$',fontsize=18)
plt.savefig(dir+os.sep+'velocityz'+str(NOISE)+'.pdf', dpi=300,transparent=True,bbox_inches='tight') 
plt.show()
plt.close('all')

