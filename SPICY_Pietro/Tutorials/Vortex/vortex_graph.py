# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 21:22:50 2021

@author: pietr
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.special as sc
import os
plt.rc('text', usetex=True)      
plt.rc('font', family='serif')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
plt.close('all')
NOISE=0.0
ppp=0.002
GAMMA=10
rc=0.1
gamma=1.256431
dir ='Result'+os.sep+'VortexImg'
if os.path.isdir(dir):
    for f in os.listdir(dir):
        os.remove(os.path.join(dir, f))
    os.rmdir(dir)
os.mkdir(dir)
cTH=rc**2/gamma
with open('Result'+os.sep+'Vortex pickles'+os.sep+'vortex_data_NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'.pkl', 'rb') as pixk:
 vortex = pickle.load(pixk)
ppp=int(ppp*1000)
NOISE=int(NOISE*100)
XSHOW1,YSHOW1=np.meshgrid(np.linspace(-0.5,0.5,100),np.linspace(-0.5,0.5,100))
XSHOW2,YSHOW2=np.meshgrid(np.linspace(-0.5,0.5,20),np.linspace(-0.5,0.5,20))
TH1=np.arctan2(YSHOW1,XSHOW1)
R1=np.sqrt(XSHOW1**2+YSHOW1**2)
VTHREAL1=GAMMA/(2*np.pi*R1)*(1-np.exp(-R1**2/cTH))
U1=vortex.extrapolate_velocities([XSHOW1,YSHOW1])
U2=vortex.extrapolate_velocities([XSHOW2,YSHOW2])
P=vortex.extrapolate_pressure([XSHOW1,YSHOW1])
DIV=vortex.extrapolate_divergence([XSHOW1,YSHOW1])
rho=1
Pc=-0.5*rho*VTHREAL1**2-rho*GAMMA**2/(4*np.pi**2*cTH)*(sc.exp1(R1**2/cTH)-sc.exp1(2*R1**2/cTH))
plt.figure(1)
plt.pcolormesh(XSHOW1,YSHOW1,np.sqrt(U1[0]**2+U1[1]**2))
plt.colorbar().set_label(r'$U[m/s]$')
plt.quiver(XSHOW2,YSHOW2,U2[0],U2[1])
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
plt.figure(2)
plt.pcolormesh(XSHOW1,YSHOW1,DIV)
plt.colorbar().set_label(r'$\nabla\cdot U[1/s]$')
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'div.pdf', dpi=300,transparent=True) 
plt.show()
plt.close()
plt.figure(3)
plt.pcolormesh(XSHOW1,YSHOW1,np.sqrt(U1[0]**2+U1[1]**2)-VTHREAL1)
plt.colorbar().set_label(r'$U-U_c[m/s]$')
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'Velocity_error_vortex'+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'.pdf', dpi=300,transparent=True) 
plt.show()
plt.figure(4)
plt.pcolormesh(XSHOW1,YSHOW1,P)
plt.colorbar().set_label(r'$P[Pa]$')
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'Pressure_vortex'+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'.pdf', dpi=300,transparent=True) 
plt.show()
plt.figure(5)
plt.pcolormesh(XSHOW1,YSHOW1,P-Pc)
plt.colorbar().set_label(r'$P-P_c[Pa]$')
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'Pressure_vortex_error'+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'.pdf', dpi=300,transparent=True) 
plt.show()
plt.figure(6)
plt.scatter(vortex.XG,vortex.YG,s=0.1,color='k')
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'Seeding_vortex'+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'.pdf', dpi=300,transparent=True) 
plt.show()
U_calc=vortex.extrapolate_velocities([vortex.XG,vortex.YG])
P_calc=vortex.extrapolate_pressure([vortex.XG,vortex.YG])
U_magn_calc=np.sqrt(U_calc[0]**2+U_calc[1]**2)
TH_scat=np.arctan2(vortex.YG,vortex.XG)
R_scat=np.sqrt(vortex.XG**2+vortex.YG**2)
U_magn_corr=GAMMA/(2*np.pi*R_scat)*(1-np.exp(-R_scat**2/cTH))
P_corr=-0.5*rho*U_magn_corr**2-rho*GAMMA**2/(4*np.pi**2*cTH)*(sc.exp1(R_scat**2/cTH)-sc.exp1(2*R_scat**2/cTH))
errP=np.linalg.norm(P_calc-P_corr)/np.linalg.norm(P_corr)
errU_magn=np.linalg.norm(U_magn_calc-U_magn_corr)/np.linalg.norm(U_magn_corr)
print('Error velocity magnitude: '+str(errU_magn*100)+' %')
print('Error pressure: '+str(errP*100)+' %')
rnd = np.random.default_rng(seed=39)
X=rnd.random(400)-0.5
Y=rnd.random(400)-0.5
U_calc=vortex.extrapolate_velocities([X,Y])
plt.figure(7)
plt.quiver(X,Y,U_calc[0],U_calc[1])
plt.gca().set_aspect('equal')
plt.xlabel(r'$x[m]$')
plt.ylabel(r'$y[m]$')
plt.savefig(dir+os.sep+'Velocity_calculated_vortex'+'NOISE='+str(NOISE)+'particlesperpixel'+str(ppp)+'1e-3'+'quiver.pdf', dpi=300,transparent=True) 
plt.show()
plt.close('all')