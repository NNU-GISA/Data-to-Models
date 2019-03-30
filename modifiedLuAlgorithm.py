# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:51:38 2019

@author: Alan
"""

from luFunctions import clock_msg,orientPCA

import numpy as np
import open3d#for pcd file io, and point normal calculation
#from sklearn import decomposition#for pca
from scipy.cluster.vq import vq, kmeans, whiten
import time
import bisect
from matplotlib import pyplot as plt#for histogram and 3d visualization
#from mpl_toolkits.mplot3d import Axes3D#for 3d visualization

class Element():
    def __init__(self,i,j,xyz,norms):
        self.x = i
        self.y = j
        self.points = xyz
        self.normals = norms


#set manual factors
p1 = 0.25;
p2 = 0.30;
nx = 50;
ny = 10;
nb = 100;

begining = time.perf_counter()
start = begining
print('\nLoading point cloud')
'''
pcd_load = open3d.read_point_cloud("..\data\Bridge1ExtraClean.pcd")
xyz_load = np.asarray(pcd_load.points)
rgb_load = np.asarray(pcd_load.colors)

start = clock_msg('Compute Point Normals',start,begining)
open3d.estimate_normals(pcd_load, search_param = open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
normals = np.asarray(pcd_load.normals)
'''

pcd_load = open3d.read_point_cloud("..\data\Bridge1_w_normals.pcd")
pcd_load = open3d.voxel_down_sample(pcd_load, voxel_size = 0.1)
#pcd_load,ind = open3d.statistical_outlier_removal(pcd_load,
#            nb_neighbors=20, std_ratio=0.5)
pcd_load,ind = open3d.radius_outlier_removal(pcd_load,
            nb_points=50, radius=0.5)
xyz_load = np.asarray(pcd_load.points)
rgb_load = np.asarray(pcd_load.colors)
normals = np.asarray(pcd_load.normals)

#Orient Bridge along x axis using PCA
start = clock_msg('Orienting point cloud along x axis',start,begining)
xyz_oriented, coef = orientPCA(xyz_load)
normals = np.matmul(normals,coef)


#slice x axis based on some delta (use 100 slices to start)
start = clock_msg('Sorting',start,begining)
index = np.argsort(xyz_oriented[:,0])
xyz = xyz_oriented[index, :]
normals = normals[index,:]

xMin = np.min(xyz[:,0])
yMin = np.min(xyz[:,1])
zMin = np.min(xyz[:,2])
xMax = np.max(xyz[:,0])
yMax = np.max(xyz[:,1])
zMax = np.max(xyz[:,2])

'''
#Cluster Global Planes
start = clock_msg('Clustering global planar surface groups',start,begining)
voxel_down_pcd = open3d.voxel_down_sample(pcd_load, voxel_size = 0.02)
vp = np.asarray(voxel_down_pcd.points)[:,]
vn = np.asarray(voxel_down_pcd.normals)[:,]

#Orient the Downsampled PointCloud
vp = np.matmul(vp,coef)
vn = np.matmul(vn,coef)


index = abs(vn[:,2])>0.99
vfp = vp[index,:]
vfn = vn[index,:]

plt.scatter(vfp[:,1],vfp[:,2])

whitened = whiten(vn[:,1:3])
'''



start = clock_msg('Slicing along X',start,begining)
deltaX = (1/nx)*(xMax-xMin)
BL = 0
xPointSlice = []
xNormSlice = []
for i in range(nx):
    BR = bisect.bisect_left(xyz[:,0],deltaX*(i+1)+xMin)
    xPointSlice.append(xyz[BL:BR,:])
    xNormSlice.append(normals[BL:BR,:])
    index = np.argsort(xPointSlice[i][:,1])
    xPointSlice[i] = xPointSlice[i][index, :]
    xNormSlice[i] = xNormSlice[i][index, :]
    BL = BR
    
start = clock_msg('Slicing along Y and populating element list',start,begining)

deltaY = (1/ny)*(yMax-yMin)
BL = 0
elementList = [[None for col in range(ny)] for row in range(nx)]
for i in range(nx):
    
    for j in range(ny):
        BR = bisect.bisect_left(xPointSlice[i][:,1],deltaY*(j+1)+yMin)
        elementList[i][j] = Element(i,j,xPointSlice[i][BL:BR,:],xNormSlice[i][BL:BR,:])
        #print("i=%d\t j=%d\t BL=%5d\t BR=%5d\t sizePoints=%5d\t"%(i,j,BL,BR,elementList[i][j].points.size))
        BL = BR
        
red = np.diag(np.divide([255, 0, 0],255))
blue = np.diag(np.divide([0, 0, 255],255))
white = np.diag(np.divide([255, 255, 255],255))
for i in range(nx):
    for j in range(ny):
        filename = "../data/globalPlanes/element_" + str(i) + "," + str(j) + "_.pcd"
        p = elementList[i][j].points
        n = elementList[i][j].normals
        index = n[:,2]>0.99
        p = p[index]
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(p)
        if (i%2==0 and j%2!=0) or (i%2!=0 and j%2==0):
            rgb = np.matmul(np.ones((len(p),3)),blue)
        else:
            rgb = np.matmul(np.ones((len(p),3)),white)
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud(filename, pcd_export)
        
#next steps are to calculate the 
        


start = clock_msg('',start,begining)

