# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 14:31:56 2021

@author: pietr
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import scipy.io
import shutil
plt.close('all')
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=18)
plt.rc('ytick',labelsize=18)
data=scipy.io.loadmat('FluentSol.mat')#extract datas from CFD ansys
X = data['x'];  Y = data['y']; # Mesh Points
P = data['p']; vx = data['vx']; vy = data['vy'] # Variables
dir = 'Result'+os.sep+'CylinderImg'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
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
circlebool=np.logical_and(np.logical_and(vx==0,vy==0),np.logical_and(Y!=0,Y!=H))
NOISE=0.1
delta=0.003
nparticles=18755
with open('Result'+os.sep+'Cylinder pickles'+os.sep+'cyl_data_NOISE='+str(NOISE)+'numberofparticles'+str(nparticles)+'number'+str(6.0)+'.pkl','rb') as pixk:
 cyl = pickle.load(pixk)
NOISE=int(NOISE*100)    
U=cyl.extrapolate_velocities([X,Y])# evaluate velocities in X and Y care the output is alist with [u,v]
## print the various errors
print('Error Velocity in the velocity X component: '+str((np.linalg.norm(U[0]-vx)+np.linalg.norm(U[1]-vy))/(np.linalg.norm(vx)+np.linalg.norm(vy))))
print('Error Velocity in the velocity Y component: '+str(np.linalg.norm(U[1]-vy)/np.linalg.norm(vy)))
print('Error Velocity in the velocity magnitude component: '+str(np.linalg.norm(np.sqrt(U[1]**2+U[0]**2)-np.sqrt(vy**2+vx**2))/np.linalg.norm(np.sqrt(vy**2+vx**2))))
print('Error Velocity in the pressure: '+str(np.linalg.norm(cyl.extrapolate_pressure([X,Y])-P)/np.linalg.norm(P)))
fig,ax=plt.subplots(figsize=(11,4.1))
area = 15  # 0 to 15 point radii
SCA=plt.scatter(X, Y, s=area, c=np.abs(P-cyl.extrapolate_pressure([X,Y])), alpha=0.9)
plt.gca().set_aspect('equal')
plt.xlabel('$x $ [m]',fontsize=18)
plt.ylabel('$y $ [m]',fontsize=18)
ax.set_xticks(np.round(np.linspace(0,1.1,5),1).tolist())
ax.set_yticks([0,0.2,0.41])
plt.title(r'$|\mathbf{p}_i-{\Phi}(\mathbf{x}_i)\mathbf{w}_p|$',fontsize=18)
plt.colorbar().set_label('$[Pa]$',fontsize=18)
plt.tight_layout()
plt.savefig(dir+os.sep+'pressure error countour cylinder.pdf',transparent=True, dpi=300)
fig,ax=plt.subplots(figsize=(11,4.1))
area = 15  # 0 to 15 point radii
SCA=plt.scatter(X, Y, s=area, c=np.abs(vx-U[0])+np.abs(vy-U[1]), alpha=0.9)
plt.gca().set_aspect('equal')
plt.xlabel('$x $ [m]',fontsize=18)
plt.ylabel('$y $ [m]',fontsize=18)
ax.set_xticks(np.round(np.linspace(0,1.1,5),1).tolist())
ax.set_yticks([0,0.2,0.41])
plt.title('$|\mathbf{u}_i-{\Phi}(\mathbf{x}_i)\mathbf{w}_u|+|\mathbf{v}_i-{\Phi}(\mathbf{x}_i)\mathbf{w}_v|$',fontsize=18)
plt.colorbar().set_label('$[m/s]$',fontsize=18)
plt.tight_layout()
plt.savefig(dir+os.sep+'velocity error countour cylinder.pdf',transparent=True, dpi=300)
##Create the logical to individuate the point in the slices
Xg=cyl.XG
Yg=cyl.YG
vxg=cyl.u
vyg=cyl.v
for theta in np.linspace(0,np.pi,3):
    if theta==0:
        logical_plot=np.logical_and(np.logical_and(X>0.2,X<0.4),np.logical_and(Y>-delta+0.2,Y<0.2+delta))
        logical_plotg=np.logical_and(np.logical_and(Xg>0.2,Xg<0.4),np.logical_and(Yg>-delta+0.2,Yg<0.2+delta))
    elif theta==np.pi:
            logical_plot=np.logical_and(np.logical_and(X>0,X<0.2),np.logical_and(Y>-delta+0.2,Y<0.2+delta))
            logical_plotg=np.logical_and(np.logical_and(Xg>0,Xg<0.2),np.logical_and(Yg>-delta+0.2,Yg<0.2+delta))
    elif theta==np.pi/2:
            logical_plot=np.logical_and(np.logical_and(Y>0.2,Y<0.4),np.logical_and(X>-delta+0.2,X<0.2+delta))
            logical_plotg=np.logical_and(np.logical_and(Yg>0.2,Yg<0.4),np.logical_and(Xg>-delta+0.2,Xg<0.2+delta))
    XXXplot=X[logical_plot]
    YYYplot=Y[logical_plot]
    RRplot=np.sqrt((XXXplot-0.2)**2+(YYYplot-0.2)**2)
    sorting=np.argsort(RRplot)
    RRplot=RRplot[sorting]/R
    mask=np.logical_and(RRplot>1,RRplot<4)
    RRplot=RRplot[mask]
    VXplot=vx[logical_plot]
    VYplot=vy[logical_plot]
    Pplot=P[logical_plot]
    VXplot=VXplot[sorting]
    VYplot=VYplot[sorting]
    Pplot=Pplot[sorting]
    VXplot=VXplot[mask]
    VYplot=VYplot[mask]
    Pplot=Pplot[mask]
    XXXplot=XXXplot[sorting]
    YYYplot=YYYplot[sorting]
    XXXplot=XXXplot[mask]
    YYYplot=YYYplot[mask]
    XXXgplot=Xg[logical_plotg]
    YYYgplot=Yg[logical_plotg]
    RRgplot=np.sqrt((XXXgplot-0.2)**2+(YYYgplot-0.2)**2)
    sortingg=np.argsort(RRgplot)
    RRgplot=RRgplot[sortingg]/R
    maskg=np.logical_and(RRgplot>1,RRgplot<4)
    RRgplot=RRgplot[maskg]
    VXgplot=vxg[logical_plotg]
    VYgplot=vyg[logical_plotg]
    VXgplot=VXgplot[sortingg]
    VYgplot=VYgplot[sortingg]
    VXgplot=VXgplot[maskg]
    VYgplot=VYgplot[maskg]
    XXXgplot=XXXgplot[sortingg]
    YYYgplot=YYYgplot[sortingg]
    XXXgplot=XXXgplot[maskg]
    YYYgplot=YYYgplot[maskg]
    V_magn=np.sqrt(VXplot**2+VYplot**2)
    Rpp=np.linspace(1,4,1000)
    Xpp=Rpp*R*np.cos(theta)+0.2
    Ypp=Rpp*R*np.sin(theta)+0.2
    Uplot=cyl.extrapolate_velocities([XXXplot,YYYplot])
    Upp=cyl.extrapolate_velocities([Xpp,Ypp])
    Press=cyl.extrapolate_pressure([Xpp,Ypp])
    Pressplot=cyl.extrapolate_pressure([XXXplot,YYYplot])
    ##plot velocities
    theta=theta*180/np.pi
    plt.figure()
    plt.plot(V_magn,RRplot,'bo',markeredgewidth=1.5,markeredgecolor='black')
    plt.plot(np.sqrt(VXgplot**2+VYgplot**2),RRgplot,'gD',markeredgewidth=1.5,markeredgecolor='black')
    plt.plot(np.sqrt(Upp[0]**2+Upp[1]**2),Rpp,'k-')
    plt.legend(['Reference data', '10\% noise','Result'])
    plt.title(r'Velocity around the cylinder at $\theta$ = '+str(np.round(theta,2))+r' $[^o]$')
    plt.ylabel(r'$r/R$')
    plt.xlabel(r'$U[m/s]$')
    plt.savefig(dir+os.sep+'Velocity error '+'theta='+str(theta)+'.pdf', dpi=300,transparent=True) 
    plt.show()
    plt.close()
    plt.figure()
    plt.plot(Pplot,RRplot,'bo',markeredgewidth=1.5,markeredgecolor='black')
    plt.plot(Press,Rpp,'k-')
    plt.legend(['Reference data','Result'])
    plt.title(r'Pressure around the cylinder at $\theta$ = '+str(np.round(theta,2))+r' $[^o]$')
    plt.ylabel(r'$r/R$')
    plt.xlabel(r'$p[Pa]$')
    plt.savefig(dir+os.sep+'Pressure error '+'theta='+str(theta)+'.pdf', dpi=300,transparent=True) 
    plt.show()
    plt.close()
Xcircle=X[circlebool]
Ycircle=Y[circlebool]
PCircle=P[circlebool]
theta=np.arctan2((Ycircle-0.2),-(Xcircle-0.2))*180/np.pi
alfa=np.linspace(-np.pi,np.pi,1000)
PCircle_comp=cyl.extrapolate_pressure([-R*np.cos(alfa)+0.2,R*np.sin(alfa)+0.2])
alfa=alfa*180/np.pi
##plot pressure around cylinder
plt.figure()
plt.plot(theta+180,PCircle,'bo',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(alfa+180,PCircle_comp,'k-')
plt.legend(['Reference data', 'Computed'])
plt.xlabel(r'$\theta [^o]$')
plt.ylabel(r'$p [Pa]$')
plt.savefig(dir+os.sep+'Pdiff'+'.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
xx=np.linspace(0.25,1.1,5)
logical2=np.zeros(len(X),dtype=bool)
logical2g=np.zeros(len(Xg),dtype=bool)
for k in xx:
    logicaltemp=np.logical_and(X<k+delta,X>k-delta)
    logicaltempg=np.logical_and(Xg<k+delta,Xg>k-delta)
    if k==0.25:
        log2=logicaltemp
        log2g=logicaltempg
    else:
     log2=np.vstack((log2,logicaltemp))
     log2g=np.vstack((log2g,logicaltempg))
    logical2=np.logical_or(logical2,logicaltemp)
    logical2g=np.logical_or(logical2g,logicaltempg)
yy=np.linspace(0,0.41,1000)

U1=cyl.extrapolate_velocities([np.ones(1000)*xx[0],yy])
U2=cyl.extrapolate_velocities([np.ones(1000)*xx[1],yy])
U3=cyl.extrapolate_velocities([np.ones(1000)*xx[2],yy])
U4=cyl.extrapolate_velocities([np.ones(1000)*xx[3],yy])
U5=cyl.extrapolate_velocities([np.ones(1000)*xx[4],yy])
xx=xx*10
plt.figure()
plt.plot(vx[log2[0]]+xx[0],Y[log2[0]],'bo',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vx[log2[1]]+xx[1],Y[log2[1]],'bo',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vx[log2[2]]+xx[2],Y[log2[2]],'bo',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vx[log2[3]]+xx[3],Y[log2[3]],'bo',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vx[log2[4]]+xx[4],Y[log2[4]],'bo',markeredgewidth=1.5,markeredgecolor='black')
plt.legend(['Reference data'])
plt.plot(vxg[log2g[0]]+xx[0],Yg[log2g[0]],'gD',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vxg[log2g[1]]+xx[1],Yg[log2g[1]],'gD',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vxg[log2g[2]]+xx[2],Yg[log2g[2]],'gD',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vxg[log2g[3]]+xx[3],Yg[log2g[3]],'gD',markeredgewidth=1.5,markeredgecolor='black')
plt.plot(vxg[log2g[4]]+xx[4],Yg[log2g[4]],'gD',markeredgewidth=1.5,markeredgecolor='black')
plt.legend(['10\% noise'])
#plt.gca().set_aspect('equal')
plt.plot(U1[0]+xx[0],yy,'k-')
plt.plot(U2[0]+xx[1],yy,'k-')
plt.plot(U3[0]+xx[2],yy,'k-')
plt.plot(U4[0]+xx[3],yy,'k-')
plt.plot(U5[0]+xx[4],yy,'k-')
plt.legend(['Results'])
plt.title(r'Velocity profiles')
plt.ylabel(r'$y[m]$')
plt.xlabel(r'$x[m]$')
plt.savefig(dir+os.sep+'Velocitywake.pdf', dpi=300,transparent=True) 
plt.show()
plt.close('all')