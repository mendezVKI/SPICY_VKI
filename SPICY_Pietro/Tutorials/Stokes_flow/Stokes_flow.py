# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 09:01:40 2021

@author: pietr
"""
import numpy as np
from SPICY.spclass import mesh_lab
import os
import shutil
import pickle
block1=False

#Physical size of the domain
D=1
D2=2
LX=2
LY=2
LZ=2
U_0=1
R=D/2
mu=1
rnd = np.random.default_rng(seed=39)
dir = 'Result'
if os.path.isdir(dir):
 shutil.rmtree(dir)
os.mkdir(dir)
dir = 'Result'+os.sep+'Stokes pickles'
os.mkdir(dir)
#Two loops to do more than one calculation with different noise od number of particles
for NOISE in np.array([0,0.05]):
    for nparticles in np.array([40000]):
        
        #Defining the domain
        Xcentersphere=0
        Ycentersphere=0
        Zcentersphere=0
        Xtemp=rnd.random(nparticles)*LX-LX/2
        Ytemp=rnd.random(nparticles)*LY-LY/2
        Ztemp=rnd.random(nparticles)*LZ-LZ/2
        Insidespherebool=(Xtemp-Xcentersphere)**2+(Ytemp-Ycentersphere)**2+(Ztemp-Zcentersphere)**2-(D/2)**2>0
        X=Xtemp[Insidespherebool]
        Y=Ytemp[Insidespherebool]
        Z=Ztemp[Insidespherebool]
        Outsidespherebool=(X-Xcentersphere)**2+(Y-Ycentersphere)**2+(Z-Zcentersphere)**2-(D2/2)**2<0
        X=X[Outsidespherebool]
        Y=Y[Outsidespherebool]
        Z=Z[Outsidespherebool]
        
        #Calculating the velocities with the Stokes flow equations
        phi=np.arctan2(Y-Ycentersphere,X-Xcentersphere)
        th=np.arctan2(np.sqrt((X-Xcentersphere)**2+(Y-Ycentersphere)**2),(Z-Zcentersphere))
        r=np.sqrt((X-Xcentersphere)**2+(Y-Ycentersphere)**2+(Z-Zcentersphere)**2)
        v_r=+U_0*(1-3/2*R/r+0.5*(R/r)**3)*np.cos(th) #theorical velocities
        v_th=-U_0*(1-3/4*R/r-0.25*(R/r)**3)*np.sin(th)
        ureal=v_r*np.cos(phi)*np.sin(th)+v_th*np.cos(phi)*np.cos(th)
        vreal=v_r*np.sin(phi)*np.sin(th)+v_th*np.cos(th)*np.sin(phi)
        wreal=v_r*np.cos(th)-v_th*np.sin(th)
        
        #Adding noise to the results
        rngu = np.random.default_rng(seed=47)
        rngv = np.random.default_rng(seed=29)
        rngw = np.random.default_rng(seed=35)
        u=ureal+(2*rngu.random(ureal.shape)-1)*NOISE*ureal
        v=vreal+(2*rngv.random(vreal.shape)-1)*NOISE*vreal
        w=wreal+(2*rngw.random(wreal.shape)-1)*NOISE*wreal
        
        #Defining mesh_lab
        stokes=mesh_lab([u,v,w],[X,Y,Z])
        
        #Defining the constrint points (outer sphere)
        NC1=90
        phicon1=np.linspace(np.pi/NC1,np.pi*(NC1-1)/NC1,NC1-2)
        RR=D2/2*np.sin(phicon1)
        passo=(np.pi*D2)/NC1
        NCperth=np.array(np.floor(RR*2*np.pi/(passo)),dtype=np.int)
        RR=RR[NCperth>=2]
        phicon1=phicon1[NCperth>=2]
        NCperth=NCperth[NCperth>=2]
        thconext=np.linspace(0,2*np.pi*(NCperth[0]-1)/NCperth[0],NCperth[0])
        phiconext=phicon1[0]*np.ones(len(thconext))
        for k in np.arange(1,len(NCperth)):
         thcon1=np.linspace(0,2*np.pi*(NCperth[k]-1)/NCperth[k],NCperth[k]-1)
         thconext=np.hstack((thconext,thcon1))
         phiconext=np.hstack((phiconext,phicon1[k]*np.ones(len(thcon1))))
        XCON1=0.5*D2*np.cos(phiconext)*np.sin(thconext)+Xcentersphere
        YCON1=0.5*D2*np.sin(phiconext)*np.sin(thconext)+Ycentersphere
        ZCON1 =0.5*D2*np.cos(thconext)+Zcentersphere
        XCON1=np.hstack((XCON1,Xcentersphere,Xcentersphere,Xcentersphere,Xcentersphere,Xcentersphere+D2/2,Xcentersphere-D2/2))
        YCON1=np.hstack((YCON1,Ycentersphere,Ycentersphere,Ycentersphere+D2/2,Ycentersphere-D2/2,Ycentersphere,Ycentersphere))
        ZCON1=np.hstack((ZCON1,Zcentersphere+D2/2,Zcentersphere-D2/2,Zcentersphere,Zcentersphere,Zcentersphere,Zcentersphere))
        
        #Defining the constrint points (inner sphere)
        NC2=60
        phicon1=np.linspace(np.pi/NC2,np.pi*(NC2-1)/NC2,NC2-2)
        RR=D/2*np.sin(phicon1)
        passo=(np.pi*D)/NC2
        NCperth=np.array(np.floor(RR*2*np.pi/(passo)),dtype=np.int)
        RR=RR[NCperth>=2]
        phicon1=phicon1[NCperth>=2]
        NCperth=NCperth[NCperth>=2]
        thconint=np.linspace(0,2*np.pi*(NCperth[0]-1)/NCperth[0],NCperth[0])
        phiconint=phicon1[0]*np.ones(len(thconint))
        for k in np.arange(1,len(NCperth)):
         thcon1=np.linspace(0,2*np.pi*(NCperth[k]-1)/NCperth[k],NCperth[k]-1)
         thconint=np.hstack((thconint,thcon1))
         phiconint=np.hstack((phiconint,phicon1[k]*np.ones(len(thcon1))))
        XCON2=0.5*D*np.cos(phiconint)*np.sin(thconint)+Xcentersphere
        YCON2=0.5*D*np.sin(phiconint)*np.sin(thconint)+Ycentersphere
        ZCON2 =0.5*D*np.cos(thconint)+Zcentersphere
        XCON2=np.hstack((XCON2,Xcentersphere,Xcentersphere,Xcentersphere,Xcentersphere,Xcentersphere+D/2,Xcentersphere-D/2))
        YCON2=np.hstack((YCON2,Ycentersphere,Ycentersphere,Ycentersphere+D/2,Ycentersphere-D/2,Ycentersphere,Ycentersphere))
        ZCON2=np.hstack((ZCON2,Zcentersphere+D/2,Zcentersphere-D/2,Zcentersphere,Zcentersphere,Zcentersphere,Zcentersphere))
        
        #Defining the constraints
        con_u=['only_div',np.zeros(len(XCON2))]
        con_v=['only_div',np.zeros(len(XCON2))]
        con_w=['only_div',np.zeros(len(XCON2))]
        CON=[con_u,con_v,con_w]
        XCON=[XCON1,XCON2]
        YCON=[YCON1,YCON2]
        ZCON=[ZCON1,ZCON2]
        
        #Define the constraint in meshlab
        stokes.velocities_constraint_definition([XCON,YCON,ZCON,CON])
        
        #Clustering the velocities
        stokes.clustering_velocities([6,10,20],cap=4.083771897425466,mincluster=[True,True,False],el=0.78)
        
        #Fit the velocities
        stokes.approximation_velocities(DIV=25,rcond=1e-13,method='fullcho')
        
        #Adding the  Dirichlet Boundary conditions
        XCON11=XCON2[-6::]
        YCON11=YCON2[-6::]
        ZCON11=ZCON2[-6::]
        phi11=np.arctan2(YCON11-Ycentersphere,XCON11-Xcentersphere)
        th11=np.arctan2(np.sqrt((XCON11-Xcentersphere)**2+(YCON11-Ycentersphere)**2),(ZCON11-Zcentersphere))
        r11=np.sqrt((XCON11-Xcentersphere)**2+(YCON11-Ycentersphere)**2+(ZCON11-Zcentersphere)**2)
        P11correct=-1.5*((mu*U_0*R/r11**2))*np.cos(th11)
        
        #Define Boundary conditions
        BCD=[P11correct,'Neumann','Neumann Wall']
        
        #Normal to boundary
        n11=np.vstack((np.cos(phi11)*np.sin(th11),np.sin(phi11)*np.sin(th11),np.cos(th11)))
        n1=np.vstack((np.cos(phiconext)*np.sin(thconext),np.sin(phiconext)*np.sin(thconext),np.cos(thconext)))
        n1=np.hstack((n1,(np.array([[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]])).T))
        n2=np.vstack((np.cos(phiconint)*np.sin(thconint),np.sin(phiconint)*np.sin(thconint),np.cos(thconint)))
        n2=np.hstack((n2,(np.array([[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]])).T))
        n=[n11,n1,n2]
        XCON=[XCON11,XCON1,XCON2]
        YCON=[YCON11,YCON1,YCON2]
        ZCON=[ZCON11,ZCON1,ZCON2]
        
        #Definition of the boundary conditions in mesh lab
        stokes.pressure_boundary_conditions(0,mu,[XCON,YCON,ZCON,BCD],n)
        
        #Clustering pressure
        stokes.clustering_pressure()
        
        #Pressure computation
        stokes.pressure_computation(rcond=1e-13,method='fullcho')

        #Save the pickle of the result
        with open(dir+os.sep+'stokes_data_NOISE='+str(NOISE)+'numberofparticles'+str(nparticles)+'.pkl', 'wb') as outp:
         pickle.dump(stokes, outp, pickle.HIGHEST_PROTOCOL)