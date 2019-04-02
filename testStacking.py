# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 19:39:50 2019

@author: Alan
"""

import numpy as np
from luFunctions import clock_msg
import time

begining = time.perf_counter()
start = begining
a = np.random.rand(1*10**8,1)


start = clock_msg("Using hstack",start,begining)
b = np.ones_like(a)
b = np.hstack((a,b))

start = clock_msg("Using empty pre-aloc",start,begining)
c = np.ones((len(a),2))
c[:,0]=a[:,0]
start = clock_msg("",start,begining)


k=0
x=0
y=0
dx=1
dy=1
nx=5
ny=5
e = eMat[0][0][0]
mergeNeighbors(k,x,y,e,dx,dy,nx,ny)


def createExportBounds(bounds, k, x, y, ny, nx):
    boundExport = np.zeros((4))
    boundExport[0] = bounds[k][0]+(bounds[k][1]-bounds[k][0])*(x/ny)
    boundExport[1] = bounds[k][0]+(bounds[k][1]-bounds[k][0])*((x+1)/ny)
    boundExport[2] = bounds[k][2]+(bounds[k][3]-bounds[k][2])*(y/nx)
    boundExport[3] = bounds[k][2]+(bounds[k][3]-bounds[k][2])*((y+1)/nx)
    return boundExport


def createEB_x(bounds, k, x, ny):
    boundExport = np.zeros((4))
    boundExport[0] = bounds[k][0]+(bounds[k][1]-bounds[k][0])*(x/ny)
    boundExport[1] = bounds[k][0]+(bounds[k][1]-bounds[k][0])*((x+1)/ny)
    boundExport[2] = bounds[k][2]+(bounds[k][3]-bounds[k][2])
    boundExport[3] = bounds[k][2]+(bounds[k][3]-bounds[k][2])
    return boundExport


def spreadNeighbors(k,x,y,e,eMat,dx,dy,nx,ny,zMin,zMax):
    #for each adjacent element
        #np.append(adjacent element surfTemp, element surf)
    print("k=%d\tx=%d\ty=%d\t"%(k,x,y))
    x_indices = np.arange(x-dx,x+dx+1)
    x_indices = x_indices[(x_indices>=0) & (x_indices<nx)]
    y_indices = np.arange(y-dy,y+dy+1)
    y_indices = y_indices[(y_indices>=0) & (y_indices<ny)]
    for i in x_indices:
        for j in y_indices:
            if not (i==x and j==y):
                #print("k=%d\ti=%d\tj=%d\t"%(k,i,j))
                eMat[k][i][j].surfTemp = np.append(eMat[k][i][j].surfTemp,e.surf)
    return eMat
def updateSurf(e, zMin, zMax):
    e.surfTemp = histCluster(e.surfTemp,zMin,zMax,e.surf)
    changed = len(e.surfTemp)
    e.surf = np.append(e.surf,e.surfTemp)
    e.surf.sort()
    return changed





xMin = np.min(surf1[:,0])
xMax = np.max(surf1[:,0])
yMin = np.min(surf1[:,1])
yMax = np.max(surf1[:,1])
zMin = np.min(surf1[:,2])
zMax = np.max(surf1[:,2])

xVal = np.linspace(xMin,xMax,100)
yVal = np.linspace(yMin,yMax,100)
sol = np.array([ 3.05844398e-05,  7.50609748e-05,  1.05210632e-02, -2.83376941e+00])
a = sol[0]
b = sol[1]
c = sol[2]
d = sol[3]

res = []
for x in range(100):
    for y in range(100):
        z=(-(a/c)*xVal[x]+(-b/c)*yVal[y]+d/c)
        if len(res)==0:
            res = np.array((xVal[x],yVal[y],z))
        else:
            res = np.vstack((res,(xVal[x],yVal[y],z)))
            
        

res = np.empty((100,3))
res[:,0] = x
res[:,1] = y
res[:,2] = z

res = curveFit(p[0][0])
k = 0
layer = 0
filename = "../data/elements/surfPCD_Plane_" + str(k) + "," + str(layer) + ".pcd"
pcd_export = open3d.PointCloud()
color = np.diag(np.divide([0, 0, 0],255))
rgb = np.matmul(np.ones((len(res),3)),color)
pcd_export.colors = open3d.Vector3dVector(rgb)
pcd_export.points = open3d.Vector3dVector(res)
open3d.write_point_cloud(filename, pcd_export)






























