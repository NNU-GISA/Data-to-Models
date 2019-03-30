# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:34:23 2019

@author: Alan
"""
import open3d
import salan_dbscan
import numpy as np
from luFunctions import clock_msg
#This file will contain functions that pertain to clustering and neighbor sharing of surfaces
#Clustering seems to have been done but ommited from the lu paper
#neighbor sharing was mentioned in lu paper but was not explained
#   Thus, both of these will be implimented from scratch using my best judgement
def xyBounds(xyz,delta):
        xMin = np.min(xyz[:,0])
        yMin = np.min(xyz[:,1])
        xMax = np.max(xyz[:,0])
        yMax = np.max(xyz[:,1])
        xLeft = xMin-(xMax-xMin)*delta
        xRight = xMax+(xMax-xMin)*delta
        yLeft = yMin-(yMax-yMin)*delta
        yRight = yMax+(yMax-yMin)*delta
        return [xLeft, xRight, yLeft, yRight]
    
def extract(data, bounds):
        res = []
        for i in range(len(bounds)):
            r1 = data[:,0]>=bounds[i,0]
            r2 = data[:,0]<=bounds[i,1]
            r3 = data[:,1]>=bounds[i,2]
            r4 = data[:,1]<=bounds[i,3]
            index = (r1 & r2 & r3 & r4)
            res.append(data[index,:])
        return res

def clusterComponents(pier, pierCap,voxel_size, plot, start, begining):
    #attempt to cluster the component point clouds (pier used for testing)
    start = clock_msg('Clustering Piers',start,begining)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pier)
    pcd = open3d.voxel_down_sample(pcd, voxel_size = 0.1)
    xyz= np.asarray(pcd.points)
    clusterPier, labelsPier = salan_dbscan.hdbscan_fun(xyz,plot)
    
    start = clock_msg('Clustering PierCaps',start,begining)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pierCap)
    pcd = open3d.voxel_down_sample(pcd, voxel_size = 0.1)
    xyz= np.asarray(pcd.points)
    clusterPierCap, labelsPierCap = salan_dbscan.hdbscan_fun(xyz,plot)
    
    #using the clustering data, find the bounding boxes for each 
    #cluster and extract those points from the FULL point set
    

    scale = 0
    bounds = np.zeros((len(clusterPier),4))
    for k in range(len(clusterPier)):
        bounds[k] = xyBounds(clusterPier[k], scale)
    cpFull = extract(pier, bounds)
    
    #cluster the pierCap next
    scale = 0
    bounds = np.zeros((len(clusterPierCap),4))
    for k in range(len(clusterPierCap)):
        bounds[k] = xyBounds(clusterPierCap[k], scale)
    cpcFull = extract(pierCap, bounds)
    
    return cpFull, cpcFull, start


