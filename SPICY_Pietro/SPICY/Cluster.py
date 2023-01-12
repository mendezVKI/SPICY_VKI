# -*- coding: utf-8 -*-
"""
Created on Fri Nov  4 11:21:02 2022

@author: Pietro.Sperotto
"""
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors
def Clusteringmethod(XG,YG,N,elongation,cap,mincluster=[False]):
    """
   This method creates clusters of the data point which is the use to find the collocation point and
   c's
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------                         
     :param XG: array
               data position in X
                        
     :param YG: array 
               data position in Y
     
     :param N: array
               number of mean point per gaussian any cell is one level
     
     :param cap:float
               cap to the shape paremeter values
     
     :param mincluster:array bool (optional)
               assign to every cluster which is smaller than N the minimum c possible
               very useful when N is small therefore could be that there are a lot of cluster of 1 2 elements
     
                        
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return X_C: array  
               collocation point X
     :return Y_C: array 
               collocation point Y
     :return c:  array
               containing shape factor
   """  
    #Dummy assigned to avoid ruining the original data
    Xdummy=np.copy(XG)
    Ydummy=np.copy(YG)
    
    # logical that check if mincluster is activated
    if np.logical_and(np.invert(mincluster[0]),len(mincluster)==1):
     mincluster=np.zeros(len(N), dtype=bool)
     #%% Calculation of the c's and collocation point
     
    #This is repeated for every layer of clustering
    for k in np.arange(0,len(N)):
     Clust=int(np.ceil(len(Xdummy)/N[k])) #number of clusters
     #Chose of the method and number of clusters
     model=MiniBatchKMeans(n_clusters=Clust, random_state=0)    
     
     #stacking
     D=np.column_stack((Xdummy,Ydummy))
     
     #obtaining the index of the points
     y_P=model.fit_predict(D)
     
     #obtaining the centers of the points
     Centers=model.cluster_centers_
     
     #Calculate the nearest neighbor for the centers
     nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(Centers)
     distances, indices = nbrs.kneighbors(Centers)
     sigma1=distances[:,1]
     
     #The collocations point for this layer are the centroids
     X_C1=Centers[:,0]
     Y_C1=Centers[:,1]
     
     #Calculate how many points there are inside every cluster
     count=np.bincount(y_P,minlength=Clust)
     sigma1[sigma1==0]=np.amax(sigma1[sigma1!=0])
     
     #If mincluster is set to True means
     #that the cluster smaller than N are automatically set to the maximum sigma
     #to be conservative
     if mincluster[k]:
      sigma1[count<N[k]]=np.amax(sigma1)
     else:
         #the same apply anyway if a cluster has only one 
         # element inside itself
      sigma1[count==1]=np.amax(sigma1)
      
     if  k==0:#If it is the first layer the arrays are initialixed
        sigma=sigma1
        X_C=np.copy(X_C1)
        Y_C=np.copy(Y_C1)
        
     else:#otherwise they are stacked
        sigma=np.hstack((sigma,sigma1))
        X_C=np.hstack((X_C,X_C1))
        Y_C=np.hstack((Y_C,Y_C1))
        
     #the dummies are updated
     Xdummy=np.copy(X_C1)
     Ydummy=np.copy(Y_C1)
     
    
    c=np.sqrt(-np.log(elongation))/(sigma)#Rule of thumb for c
    c[c>cap]=cap#cap because sometimes unusually large c appear
    return X_C,Y_C,c

def Clusteringmethod3D(XG,YG,ZG,N,elongation,cap,mincluster=[False]):
    """
   This method creates clusters of the data point which is the use to find the collocation point and
   c's
   
   ----------------------------------------------------------------------------------------------------------------
     Parameters
     ----------                         
     :param XG: array
               data position in X
                        
     :param YG: array 
               data position in Y

     :param ZG: array 
               data position in Z
     
     :param N: array
               number of mean point per gaussian any cell is one level
     
     :param cap:float 
               cap to the shape paremeter values
     
     :param mincluster:array bool (optional)
               assign to every cluster which is smaller than N the minimum c possible
               very useful when N is small therefore could be that there are a lot of cluster of 1 2 elements
     
                        
     ----------------------------------------------------------------------------------------------------------------
     Returns
     -------

     :return X_C: array  
               collocation point X
     :return Y_C: array 
               collocation point Y
     :return Z_C: array 
               collocation point Z
     :return c:  array
               containing shape factor
   """  
    #Dummy assigned to avoid ruining the original data
    Xdummy=np.copy(XG)
    Ydummy=np.copy(YG)
    Zdummy=np.copy(ZG)
    
    # logical that check if mincluster is activated
    if np.logical_and(np.invert(mincluster[0]),len(mincluster)==1):
     mincluster=np.zeros(len(N), dtype=bool)
     
    #%% Calculation of the c's and collocation point
    #This is repeated for every layer of clustering
    for k in np.arange(0,len(N)):
     Clust=int(np.ceil(len(Xdummy)/N[k])) #number of clusters
     model=MiniBatchKMeans(n_clusters=Clust, random_state=0)    
     D=np.column_stack((Xdummy,Ydummy,Zdummy))#stacking
     y_P=model.fit_predict(D)#obtaining the index of the points
     Centers=model.cluster_centers_#obtaining the centers of the points
     nbrs = NearestNeighbors(n_neighbors=2,algorithm='ball_tree',n_jobs=8).fit(Centers)#Calculate the nearest neighbor for the centers
     distances, indices = nbrs.kneighbors(Centers)
     sigma1=distances[:,1]#The distance with the nearest neighboor is sigma
     X_C1=Centers[:,0]#The collocations point for this layer are the centroids
     Y_C1=Centers[:,1]
     Z_C1=Centers[:,2]
     count=np.bincount(y_P,minlength=Clust)#Calculate how many element there are inside every cluster
     sigma1[sigma1==0]=np.amax(sigma1[sigma1!=0])
     if mincluster[k]:
      sigma1[count<N[k]]=np.amax(sigma1)#If mincluster is set to True means
      #that the cluster smaller than N are automatically set to the maximum sigma
      #to be conservative
     else:
      sigma1[count==1]=np.amax(sigma1)#the same apply anyway if a cluster has only one 
      # element inside itself
     if  k==0:#If it is the first layer the arrays are initialixed
        sigma=sigma1
        X_C=np.copy(X_C1)
        Y_C=np.copy(Y_C1)
        Z_C=np.copy(Z_C1)
        
     else:#otherwise they are stacked
        sigma=np.hstack((sigma,sigma1))
        X_C=np.hstack((X_C,X_C1))
        Y_C=np.hstack((Y_C,Y_C1))
        Z_C=np.hstack((Z_C,Z_C1))
        
     Xdummy=np.copy(X_C1)#the dummies are updated
     Ydummy=np.copy(Y_C1)
     Zdummy=np.copy(Z_C1)
    c=np.sqrt(-np.log(elongation))/(sigma)#Rule of thumb for c
    c[c>cap]=cap#cap because sometimes unusually large c appear
    return X_C,Y_C,Z_C,c