#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 10 13:06:27 2022

@author: sperotto
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
plt.close('all')
#Extract the data
A=np.loadtxt('Anistotropic data/stats_averages/stats_averages_z_1_8.text')
C=np.loadtxt('Anistotropic data/stats_rms/stats_rms_z_1_8.text')
x=A[:,0]
y=A[:,1]
u=A[:,3]
v=A[:,4]
w=A[:,5]
p=A[:,6]
XX,YY=np.meshgrid(np.linspace(0,14,100),np.linspace(-2,0,100))
dir = 'Result'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
dir = 'Result'+os.sep+'BFS DNS plot'
os.mkdir(dir)
#Interpolate the data
from scipy.interpolate import griddata as gd
pplot=gd((x,y), p, (XX,YY), method='cubic')
uplot=gd((x,y), u, (XX,YY), method='cubic')
vplot=gd((x,y), v, (XX,YY), method='cubic')
wplot=gd((x,y), w, (XX,YY), method='cubic')
TKEplot=gd((x,y), 0.5*(C[:,3]**2+C[:,4]**2+C[:,5]**2), (XX,YY), method='cubic')
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=16)
plt.rc('ytick',labelsize=16)
plt.close('all')
YY=YY+2
#plot data
plt.figure(1,figsize=(5,2))
plt.pcolormesh(XX,YY,uplot,vmax=np.amax(uplot),vmin=np.amin(uplot))
plt.colorbar()
plt.title(r'Velocity $\overline{u}[m/s]$ DNS',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.xlim([0,14])
plt.ylim([0,2])
plt.tight_layout()
plt.savefig(dir+os.sep+'u_real.pdf',dpi=500)
plt.show()
plt.figure(2,figsize=(5,2))
plt.pcolormesh(XX,YY,vplot,vmax=np.amax(vplot),vmin=np.amin(vplot))
plt.colorbar()
plt.title(r'Velocity $\overline{v}[m/s]$ DNS',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'v_real.pdf',dpi=500)
plt.show()
plt.figure(3,figsize=(5,2))
plt.pcolormesh(XX,YY,wplot,vmax=np.amax(wplot),vmin=np.amin(wplot))
plt.colorbar()
plt.title(r'Velocity $\overline{w}[m/s]$ DNS',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'w_real.pdf',dpi=500)
plt.show()
plt.figure(4,figsize=(5,2))
plt.pcolormesh(XX,YY,pplot,vmax=np.amax(pplot),vmin=np.amin(pplot))
plt.colorbar()
plt.title(r'Pressure $\overline{p}[Pa]$ DNS',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'p_real.pdf',dpi=500)
plt.show()
plt.figure(5,figsize=(5,2))
plt.pcolormesh(XX,YY,TKEplot,vmax=np.amax(TKEplot),vmin=0)
plt.colorbar()
plt.title(r'$k[m^2/s^2]$ DNS',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.savefig(dir+os.sep+'TKE_real.pdf',dpi=500)
plt.show()
plt.figure(6,figsize=(5,2))
plt.pcolormesh(XX,YY,np.sqrt(uplot**2+vplot**2),vmax=np.amax(np.sqrt(uplot**2+vplot**2)),vmin=0)
plt.colorbar()
plt.title(r'In plane velocity components $[m/s]$ DNS',fontsize=16)
plt.xlabel(r'$x[cm]$',fontsize=16)
plt.ylabel(r'$y[cm]$',fontsize=16)
plt.tight_layout()
plt.quiver(XX[::4,::4],YY[::4,::4],uplot[::4,::4],vplot[::4,::4],color='white')
plt.savefig(dir+os.sep+'uv_real.pdf',dpi=500)
plt.show()

#logical to cut the ROI
logic=np.logical_and(y<0,x>0)
logic=np.logical_and(logic,x<14)
logic=np.logical_and(logic,y>-2)

#Starting the dataset construction
name_array=['1_39','1_6','1_8','2_01','2_21']
xfin=np.array([])
yfin=np.array([])
zfin=np.array([])
ufin=np.array([])
vfin=np.array([])
wfin=np.array([])
pfin=np.array([])
RSXXfin=np.array([])
RSXYfin=np.array([])
RSXZfin=np.array([])
RSYYfin=np.array([])
RSYZfin=np.array([])
RSZZfin=np.array([])

#In each z plane of the data we extract the values in the ROI and we stack them
for k in name_array:
    A=np.loadtxt('Anistotropic data/stats_averages/stats_averages_z_'+k+'.text')
    B=np.loadtxt('Anistotropic data/stats_RS/stats_RS_z_'+k+'.text')
    C=np.loadtxt('Anistotropic data/stats_rms/stats_rms_z_'+k+'.text')
    xfin=np.hstack((xfin,A[:,0][logic]))
    yfin=np.hstack((yfin,A[:,1][logic]))
    zfin=np.hstack((zfin,A[:,2][logic]))
    ufin=np.hstack((ufin,A[:,3][logic]))
    vfin=np.hstack((vfin,A[:,4][logic]))
    wfin=np.hstack((wfin,A[:,5][logic]))
    pfin=np.hstack((pfin,A[:,6][logic]))
    RSXYfin=np.hstack((RSXYfin,B[:,3][logic]))
    RSYZfin=np.hstack((RSYZfin,B[:,5][logic]))
    RSXZfin=np.hstack((RSXZfin,B[:,4][logic]))
    RSXXfin=np.hstack((RSXXfin,(C[:,3][logic])**2))
    RSYYfin=np.hstack((RSYYfin,(C[:,4][logic])**2))
    RSZZfin=np.hstack((RSZZfin,(C[:,5][logic])**2))
    
#Adapting the size
yfin=(yfin+2)/10
xfin=(xfin)/10
zfin=(zfin-np.mean(zfin))/10

#Cycle to obtain a more uniform distrution of the seeding
from numpy.random import default_rng
rng = default_rng(seed=30)
n=20
N=8400
xfin1=np.array([])
yfin1=np.array([])
zfin1=np.array([])
ufin1=np.array([])
vfin1=np.array([])
wfin1=np.array([])
RSXXfin1=np.array([])
RSXYfin1=np.array([])
RSXZfin1=np.array([])
RSYYfin1=np.array([])
RSZZfin1=np.array([])
RSYZfin1=np.array([])
pfin1=np.array([])
linx=np.linspace(np.amin(xfin),np.amax(xfin),n)
liny=np.linspace(np.amin(yfin),np.amax(yfin),n)
Nsub=np.int32(N/((n-1)**2))+1
for k in np.arange(n-1):
    for j in np.arange(n-1):
        for i in np.unique(zfin):
            logico=np.logical_and(xfin>linx[k],xfin<linx[k+1])
            logico=np.logical_and(logico,yfin>liny[j])
            logico=np.logical_and(logico,yfin<liny[j+1])
            logico=np.logical_and(logico,i==zfin)
            xfint=xfin[logico]
            yfint=yfin[logico]
            zfint=zfin[logico]
            ufint=ufin[logico]
            vfint=vfin[logico]
            wfint=wfin[logico]
            pfint=pfin[logico]
            RSXYfint=RSXYfin[logico]
            RSYZfint=RSYZfin[logico]
            RSXZfint=RSXZfin[logico]
            RSXXfint=RSXXfin[logico]
            RSYYfint=RSYYfin[logico]
            RSZZfint=RSZZfin[logico]
            if Nsub<len(xfint):
             logic=rng.choice(np.arange(len(xfint)), Nsub, replace=False)
            else:
                logic=np.ones(len(xfint))
                logic=logic.astype(np.bool)
                
            xfin1=np.hstack((xfin1,xfint[logic]))
            yfin1=np.hstack((yfin1,yfint[logic]))
            zfin1=np.hstack((zfin1,zfint[logic]))
            ufin1=np.hstack((ufin1,ufint[logic]))
            vfin1=np.hstack((vfin1,vfint[logic]))
            wfin1=np.hstack((wfin1,wfint[logic]))
            RSXYfin1=np.hstack((RSXYfin1,RSXYfint[logic]))
            RSXXfin1=np.hstack((RSXXfin1,RSXXfint[logic]))
            RSYYfin1=np.hstack((RSYYfin1,RSYYfint[logic]))
            RSXZfin1=np.hstack((RSXZfin1,RSXZfint[logic]))
            RSZZfin1=np.hstack((RSZZfin1,RSZZfint[logic]))
            RSYZfin1=np.hstack((RSYZfin1,RSYZfint[logic]))
            pfin1=np.hstack((pfin1,pfint[logic]))
logic=rng.choice(np.arange(len(xfin1)), N, replace=False)
dir = 'Result'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
dir = 'Result'+os.sep+'Cylinder pickles'
os.mkdir(dir)
#Save data
np.savetxt('Anistotropic data/database.txt',[xfin1[logic],yfin1[logic],zfin1[logic],ufin1[logic],vfin1[logic],wfin1[logic],pfin1[logic],RSXXfin1[logic],RSXYfin1[logic],RSXZfin1[logic],RSYYfin1[logic],RSYZfin1[logic],RSZZfin1[logic]])
plt.close('all')