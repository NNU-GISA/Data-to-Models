# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 15:22:34 2019

@author: Alan
"""
from luFunctions import clock_msg
from alanFunctions import xyBounds
import salan_dbscan
import numpy as np
import open3d
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
import bisect
import time
from scipy.optimize import leastsq
import scipy.linalg
from mpl_toolkits.mplot3d import Axes3D

global color, debug, dz, info
debug = False
info = True

class Element():
    def __init__(self,k,i,j,xyz,norms,color,zMin,zMax,bounds):
        self.k = k
        self.x = i
        self.y = j
        self.points = xyz
        self.normals = norms
        self.color = color
        self.zMin = zMin
        self.zMax = zMax
        self.bounds = bounds
        
        #self.clusteredSurfPoints[0] will contain all normals who's corresponding xyz 
        #location lies on the first surface
        self.clusteredSurfPoints = []
        self.surfPoints = []
        #Holds an array of z values of the lower bound of planar surfaces
        
        self.surf = []
        self.surfOriginal = []
        #Holds surf values stolen from adjacent nodes that did not exist 
        #in the elements area and have not been clustered yet
        self.surfTemp = []
        
        
        self.empty = False
        if xyz.size == 0:
            self.empty = True
        #self.clusterSurf()
        
        
    #create the original surfaces using only the information in the given element
    def clusterSurf(self):
        if not self.empty:
            index = self.normals[:,2]>0.99
            self.surfPoints = self.points[index,:]
            index2 = self.surfPoints[:,2]>(0.5*(self.zMax-self.zMin)+self.zMin)
            self.surfPoints = self.surfPoints[index2,:]
            nb = 10
            hist = np.histogram(self.surfPoints[:,2],range=(0.5*(self.zMax-self.zMin)+self.zMin,self.zMax), bins=nb)
            ind = hist[0]>300*(50/nb)
            pos = np.where(ind)[0]
            counter = 1
            j = 0
            while 1:
                if j+1 >= len(pos):
                    break
                #if adjacent box is non zero, delete the box to the right and look again
                if (pos[j+1]==pos[j]+counter):
                    counter += 1
                    pos = np.delete(pos,j+1)
                #otherwise look to next box
                else:
                    j+=1
                    counter = 1
            for k in range(len(pos)):
                self.surf.append(hist[1][pos[k]])
        
            
            
            
    def export(self,k,color):
        #print("I'm not done yet")
        filename = "../data/elements/cluster" + str(k) + "_" + str(self.x) + "," + str(self.y) + ".pcd"
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(self.points)
        rgb = np.matmul(np.ones((len(self.points),3)),np.diag(color))
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud(filename, pcd_export)
        
def fitPlane(data):
    XYZ = data.T
    c = 0.001
    p0 = [0.0001, 0.0001, c, c*data[0,2]]#guess that the solution is a flat plane through first data point
    def f_min(X,p):
        print(p)
        plane_xyz = p[0:3]#data points x,y,z
        distance = (plane_xyz*X.T).sum(axis=1) + p[3]
        return distance / np.linalg.norm(plane_xyz)
    
    def residuals(params, signal, X):
        return f_min(X, params)
    
    sol = leastsq(residuals, p0, args=(None, XYZ))[0]
    print("Solution: ", sol)
    print("Old Error: ", (f_min(XYZ, p0)**2).sum())
    print("New Error: ", (f_min(XYZ, sol)**2).sum())
    return sol

def curveFit(data):
    #offset = dz*0.005
    plot = False
    xMin = np.min(data[:,0])
    xMax = np.max(data[:,0])
    yMin = np.min(data[:,1])
    yMax = np.max(data[:,1])
    #zMin = np.min(data[:,2])
    #zMax = np.max(data[:,2])
    X,Y = np.meshgrid(np.arange(xMin, xMax, 0.1), np.arange(yMin, yMax, 0.1))
    
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    #C[2] = C[2] - offset
    Z = C[0]*X + C[1]*Y + C[2]
    res = []
    for i in range(len(X[0,:])):
        for j in range(len(Y)):
            if len(res)==0:
                res = np.array((X[0,i],Y[j,0],Z[j,i]))
            else:
                res = np.vstack((res,(X[0,i],Y[j,0],Z[j,i])))
    if plot:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
        ax.scatter(data[:,0], data[:,1], data[:,2], c='r', s=50)
        plt.xlabel('X')
        plt.ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')
        ax.axis('tight')
        plt.show()
    return res, C

def curveFitQuad(data):
    A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])
    return C

def curveFitLinear(data):
    A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
    C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])    # coefficients
    return C

#given the coef for the equation of a plane in 3d and a data set
    #find and return the zMin and zMax within the data range
def findZBoundVals(C,data):
    xMin = np.min(data[:,0])
    xMax = np.max(data[:,0])
    yMin = np.min(data[:,1])
    yMax = np.max(data[:,1])
    def zfun(X,Y,C):
        return C[0]*X + C[1]*Y + C[2]
    p1 = zfun(xMin,yMin,C)
    p2 = zfun(xMin,yMax,C)
    p3 = zfun(xMax,yMin,C)
    p4 = zfun(xMax,yMax,C)
    possibilities = [p1,p2,p3,p4]
    mini = np.min(possibilities)
    maxi = np.max(possibilities)
    delta = maxi-mini
    
    return [mini-delta*0.5, maxi+delta*0.5]


def findZIndex(vals, data):

    index = []
    for d in range(len(vals)):
        index.append(bisect.bisect_left(data[:,2],vals[d]))
    return index


def extractIndex(data, bounds):
    r1 = data[:,0]>=bounds[0]
    r2 = data[:,0]<=bounds[1]
    r3 = data[:,1]>=bounds[2]
    r4 = data[:,1]<=bounds[3]
    index = (r1 & r2 & r3 & r4)
    return index

def createPosLeft(pos):
    posLeft = pos
    counter = 1
    j = 0
    while 1:
        if j+1 >= len(posLeft):
            break
        #if adjacent box is non zero, delete the box to the right and look again
        if (posLeft[j+1]==posLeft[j]+counter):
            counter += 1
            posLeft = np.delete(posLeft,j+1)
        #otherwise look to next box
        else:
            j+=1
            counter = 1
    return posLeft
def createPosRight(pos):
    posRight = pos
    counter = 1
    j = len(posRight)-1
    while 1:
        if j-1 < 0:
            break
        #if adjacent box is non zero, delete the box to the right and look again
        if (posRight[j-1]==posRight[j]-counter):
            posRight = np.delete(posRight,j-1)
            counter += 1
            j-=1
        #otherwise look to next box
        else:
            counter = 1
            j-=1
    return posRight
def histCluster(points,zMin,zMax,surf):
    
    res = []
    nb = 10
    dz = zMax-zMin
    if debug: print("dz: %3.3f, zMin: %3.3f, zMax: %3.3f"%(dz,zMin,zMax))
    hist = np.histogram(points,range=(0.5*(dz)+zMin,zMax), bins=nb)
    
    #hist = plt.hist(points,range=(0.5*(dz)+zMin,zMax), bins=nb)
    ind = hist[0]>0
    pos = np.where(ind)[0]
    posRight = createPosRight(pos)
    posLeft = createPosLeft(pos)
    
    #for each
    if debug:
        print("Pos= " + str(pos))
        print("PosLeft= " + str(posLeft))
        print("PosRight= " + str(posRight))
    for k in range(len(posLeft)):
        left = hist[1][posLeft[k]]
        right = hist[1][posRight[k]]
        if left==right:
            right = hist[1][posRight[k]+1]
        rb = ((left,right))
        index = (points>=rb[0]) & (points<=rb[1])
        pSub = points[index]
        
        #if {e.surf} in {pSub} -> continue
        #if the upper or lower edge of the surface are 
        index2 = ((surf>left-dz/10)&(surf<right+dz/10))
        if debug:
            print("Left=%3.3f\tRight=%3.3f\t"%(left,right))
            print("psub = " + str(pSub))
            print("index = " + str(index))
            print("index2 = " + str(index2))
        
        if np.any(index2):
            if debug: print("Continuing because surface already exists")
            continue
        if len(pSub)==0:
            if debug: print("Continuing because no points found in surfTemp to append")
            continue
        #print(pSub)
        result = (np.max(pSub)-np.min(pSub))/2+np.min(pSub)
        if debug: print("Appending %3.3f to surf"%result)
        res.append(result)
    return res
        
    
    #return a list of z values for the lower bound of each cluster

def spreadNeighbors(k,x,y,e,eMat,dx,dy,nx,ny,zMin,zMax):
    #for each adjacent element
        #np.append(adjacent element surfTemp, element surf)
    #print("k=%d\tx=%d\ty=%d\t"%(k,x,y))
    x_indices = np.arange(x-dx,x+dx+1)
    x_indices = x_indices[(x_indices>=0) & (x_indices<nx)]
    y_indices = np.arange(y-dy,y+dy+1)
    y_indices = y_indices[(y_indices>=0) & (y_indices<ny)]
    for i in x_indices:
        for j in y_indices:
            if not (i==x and j==y):
                #print("k=%d\ti=%d\tj=%d\t"%(k,i,j))
                eMat[k][i][j].surfTemp = np.append(eMat[k][i][j].surfTemp,e.surf)
def updateSurf(e, zMin, zMax):
    if debug:
        print("*"*100)
        print("e.surf = " +str(e.surf))
        print("e.surfTemp = " +str(e.surfTemp))
    e.surfTemp = histCluster(e.surfTemp,zMin,zMax,e.surf)
    changed = len(e.surfTemp)
    e.surf = np.append(e.surf,e.surfTemp)
    e.surf.sort()
    return changed
def extractSurfacePoints(eMat,cluster,nx,ny):
    p = [[[] for i in range(len(eMat[0][0][0].surf))] for i in range(len(cluster))]
    for k in range(len(cluster)):
        for i in range(ny):
            for j in range(nx):
                #print("k=%d\ti=%d\tj=%d\t"%(k,i,j))
                e = eMat[k][i][j]
                bounds = eMat[k][i][j].bounds
                cx = (bounds[1]-bounds[0])/2+bounds[0]
                cy = (bounds[3]-bounds[2])/2+bounds[2]
                for s in range(len(e.surf)):
                    if len(e.surfOriginal) > 0:
                        if e.surf[s] in e.surfOriginal:
                            res = np.array((cx,cy,e.surf[s]))
                            if len(p[k][s])==0:
                                p[k][s] = res
                            else:
                                p[k][s] = np.vstack((p[k][s],res))
    return p

def segmentNorms(eMat, cluster, nx, ny):
    #for each element
    numSurf = len(eMat[0][0][0].surf)
    surfPointsCombined = [[[] for i in range(numSurf)] for i in range(len(cluster))]
    for k in range(len(cluster)):
        flag = False
        for i in range(ny):
            for j in range(nx):
                #print("k=%d\ti=%d\tj=%d\t"%(k,i,j))
                e = eMat[k][i][j]
                if len(e.surfPoints)==0:
                    #print("No surface points to segment, skipping.")
                    continue
                e.surfPoints = e.surfPoints[np.argsort(e.surfPoints[:,2]), :]
                divider = []#dividing value
                for s in range(len(e.surf)-1):
                    divider.append((e.surf[s]+e.surf[s+1])/2)
                BL = 0#BL, BR = dividing index
                for d in range(len(divider)):
                    BR = bisect.bisect_left(e.surfPoints[:,2],divider[d])
                    #print("BL=%d\tBR=%d\td=%d\t"%(BL,BR,divider[d]))
                    e.clusteredSurfPoints.append(e.surfPoints[BL:BR,:])
                    BL = BR
                if len(divider)==0:
                    BR=0
                e.clusteredSurfPoints.append(e.surfPoints[BR:,:])

                for noti in range(numSurf):
                    if not flag:
                        #print("k=%d\ti=%d\tj=%d\tnoti=%d\t"%(k,i,j,noti))
                        surfPointsCombined[k][noti]=e.clusteredSurfPoints[noti]
                    else:
                        surfPointsCombined[k][noti] = np.vstack((surfPointsCombined[k][noti],e.clusteredSurfPoints[noti]))
                flag = True

    return surfPointsCombined
                    
                
        
                    
'''
def mergeNeighbors(k,x,y,e,eMat,dx,dy,nx,ny,zMin,zMax):
    e.surfTemp = []
    x_indices = np.arange(x-dx,x+dx+1)
    x_indices = x_indices[(x_indices>=0) & (x_indices<nx)]
    y_indices = np.arange(y-dy,y+dy+1)
    y_indices = y_indices[(y_indices>=0) & (y_indices<ny)]
    for i in x_indices:
        for j in y_indices:
            if not (i==x and j==y):
                #print("k=%d\ti=%d\tj=%d\t"%(k,i,j))
                e.surfTemp = np.append(e.surfTemp,eMat[k][i][j].surf)
                #e.surfTemp.append(eMat[k][i][j].surf)
            
    #print("k=%d\tx=%d\ty=%d\t"%(k,x,y))
    #print("e.surf before:" + str(e.surf))
    #print("e.surfTemp before:" + str(e.surfTemp))
    
    e.surfTemp = histCluster(e.surfTemp,zMin,zMax,e.surf)
    #print("e.surfTemp after:" + str(e.surfTemp))
    e.surf = np.append(e.surf,e.surfTemp)
    e.surf.sort()
    return e.surf
    
    #e.surfTemp now has most likely multiples of the surfaces missing originally
    #how to cluster them...
'''
def createSurfPCD(nx,ny,eMat, cluster):    
    flag = False       
    for k in range(len(cluster)):
        for x in range(ny):
            for y in range(nx):
                surf = eMat[k][x][y].surf
                bounds = eMat[k][x][y].bounds
                cx = (bounds[1]-bounds[0])/2+bounds[0]
                cy = (bounds[3]-bounds[2])/2+bounds[2]
                if not flag:
                    for n in range(len(surf)):
                        surfPCD =  np.array((cx,cy,surf[n]))
                        flag = True
                else:
                    for n in range(len(surf)):
                        surfPCD = np.vstack((surfPCD,np.array((cx,cy,surf[n]))))
                
                #if len(surf)==1:
                    #print("x=%d\tx=%d\ty=%d\t"%(k,x,y),end="")
                    #print(surf)
    return surfPCD

def createSurfTempPCD(nx,ny,eMat, cluster):
    flag = False       
    for k in range(len(cluster)):
        for x in range(ny):
            for y in range(nx):
                surf = eMat[k][x][y].surfTemp
                bounds = eMat[k][x][y].bounds
                cx = (bounds[1]-bounds[0])/2+bounds[0]
                cy = (bounds[3]-bounds[2])/2+bounds[2]
                if not flag:
                    for n in range(len(surf)):
                        surfPCD =  np.array((cx,cy,surf[n]))
                        flag = True
                else:
                    for n in range(len(surf)):
                        surfPCD = np.vstack((surfPCD,np.array((cx,cy,surf[n]))))
                
                #if len(surf)==1:
                    #print("x=%d\tx=%d\ty=%d\t"%(k,x,y),end="")
                    #print(surf)
    return surfPCD
def exportComponents(deck,pierCap, pier):
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    green = np.diag(np.divide([0, 255, 0],255))
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(deck)
    rgb = np.matmul(np.ones((len(deck),3)),red)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/elements/deck.pcd", pcd_export)
    if pierCap.size!=0:
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(pierCap)
        rgb = np.matmul(np.ones((len(pierCap),3)),green)
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud("../data/elements/pierCap.pcd", pcd_export)
        
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(pier)
    rgb = np.matmul(np.ones((len(pier),3)),blue)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/elements/pier.pcd", pcd_export)
    return 1

def exportComponents2(deck,pierCap, pier):
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    green = np.diag(np.divide([0, 255, 0],255))
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(deck)
    rgb = np.matmul(np.ones((len(deck),3)),red)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/elements/deck2.pcd", pcd_export)
    
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(pierCap)
    rgb = np.matmul(np.ones((len(pierCap),3)),green)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/elements/pierCap2.pcd", pcd_export)
    
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(pier)
    rgb = np.matmul(np.ones((len(pier),3)),blue)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/elements/pier2.pcd", pcd_export)
    return 1

def exportSubComponents(pcdI, ppcI):
    color = np.diag(np.divide([255, 0, 255],255))
    for k in range(3):
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(pcdI[k])
        rgb = np.matmul(np.ones((len(pcdI[k]),3)),color)
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud("../data/elements/pcdI"+str(k)+".pcd", pcd_export)
        
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(ppcI[k])
        rgb = np.matmul(np.ones((len(ppcI[k]),3)),color)
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud("../data/elements/ppcI"+str(k)+".pcd", pcd_export)

    return 1

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


'''
begining = time.perf_counter()
start = begining
write = True
pierArea = np.load("pierArea.npy")
'''

def pierAreaSegmentation(pierArea,begining,start,write):   
    print('\n*Loading point cloud')
    #Note: this nx and ny are really nx' and ny'. 
    #They correspond to number of subdivisions per "Pier Cluster"
    nx = 5
    ny = 5
        
    pierSize = 0
    for i in range(len(pierArea)):
        pierSize += len(pierArea[i])
        
    bigPierArea = np.empty((pierSize,3))
    pos = 0
    nex = 0
    for i in range(len(pierArea)):
        nex = len(pierArea[i])
        for j in range(len(pierArea[i])):
            bigPierArea[pos+j,:] = pierArea[i][j,:]
        pos += nex
    
    dz = np.max(bigPierArea[:,2])-np.min(bigPierArea[:,2])
        
    start = clock_msg('*Compute Normals',start,begining)
    
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(bigPierArea)
    open3d.estimate_normals(pcd, search_param = open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
    normals = np.asarray(pcd.normals)
        
    start = clock_msg('*Cluster by pier group and populate element objects',start,begining)
    
    pcd = open3d.voxel_down_sample(pcd, voxel_size = 0.2)
    xyz= np.asarray(pcd.points)
    plot = False
    cluster, labels = salan_dbscan.hdbscan_fun(xyz,plot)
    
    scale = 0
    bounds = np.zeros((len(cluster),4))
    for k in range(len(cluster)):
        bounds[k] = xyBounds(cluster[k], scale)
    
        
    #Now for each cluster we have: bounds, points, normals
    #Next step: Slice into cubes
    #start = clock_msg('*Populate the element objects',start,begining)
    eMat = []
    pierAreaSub = []
    normSub = []
    #pierAreaSubX = []
    #normSubX= []
    plot = False
    if plot:
        fig, ax = plt.subplots()
    for k in range(len(cluster)):
        color = iter(cm.rainbow(np.linspace(0,1,nx*ny)))
        e = [[[] for x in range(nx)] for x in range(ny)]
        zMin = np.min(cluster[k][:,2])
        zMax = np.max(cluster[k][:,2])
        #boundExport = [[np.zeros(4) for x in range(nx)] for x in range(ny)]
        #DeltaX_length = 
        index = extractIndex(bigPierArea,bounds[k])
        pierAreaSub.append(bigPierArea[index,:])
        normSub.append(normals[index,:])
        for x in range(ny):
            #Consider slicing in the x direction before cubing the data
            #time to beat: 1.73 seconds
            #BEx = createEB_x(bounds, k, x, ny)
            #index = extractIndex(bigPierArea,BEx)
            #pierAreaSubX.append(bigPierArea[index,:])
            #normSubX.append(normals[index,:])
            for y in range(nx):
                #print("k=%d\tx=%d\ty=%d\t"%(k,x,y))
                boundExport = createExportBounds(bounds, k, x, y, ny, nx)
                index = extractIndex(pierAreaSub[k], boundExport)
                pointsExport = pierAreaSub[k][index,:]
                normalsExport = normSub[k][index,:]
                e[x][y] = Element(k,x,y,pointsExport,normalsExport,next(color)[0:3],zMin,zMax,boundExport)
                if plot:
                    ax.vlines(boundExport[0],ymin=boundExport[2],ymax=boundExport[3])
                    ax.vlines(boundExport[1],ymin=boundExport[2],ymax=boundExport[3])
                    ax.hlines(boundExport[2],xmin=boundExport[0],xmax=boundExport[1])
                    ax.hlines(boundExport[3],xmin=boundExport[0],xmax=boundExport[1])
        eMat.append(e)
        
    #start = clock_msg('*Cluster Element Surfaces',start,begining)
    
    for k in range(len(cluster)):
        surfOrigCount = []
        for x in range(ny):
            for y in range(nx):
                eMat[k][x][y].clusterSurf()
                eMat[k][x][y].surfOriginal = eMat[k][x][y].surf
                eMat[k][x][y].surfOriginal.sort()
                numSurf = len(eMat[k][x][y].surfOriginal)
                if numSurf >= len(surfOrigCount):
                    for i in range(numSurf-len(surfOrigCount)+1):
                        surfOrigCount.append(0)
                surfOrigCount[numSurf]+=1
                #print(surfPCD)
        if info:
            print("In cluster %d the elements per surface count are as follows:"%k)
            print("numElement / numSurf: ",end="")
            for c in range(len(surfOrigCount)):
                print("(%d/ %d)\t"%(surfOrigCount[c],c), end = "")
            print()
    
        
        
    
    
    
    
     
      
    surfPCD = createSurfPCD(nx,ny,eMat, cluster)
    
    #Accept surfaces from adjacent dx,dy nodes (adjacent 8 squares if 1,1)
    dx=2
    dy=2
    
    start = clock_msg('*Spread surfaces to neighbors',start,begining)
    for iterrations in range(20):
        
        changed = 0
        for k in range(len(cluster)):
            for x in range(ny):
                for y in range(nx):
                    #print("k=%d\tx=%d\ty=%d\t"%(k,x,y))
                    zMin = np.min(cluster[k][:,2])
                    zMax = np.max(cluster[k][:,2])
                    spreadNeighbors(k,x,y,eMat[k][x][y],eMat,dx,dy,nx,ny,zMin,zMax)
        for k in range(len(cluster)):
            for x in range(ny):
                for y in range(nx):
                    #print("k=%d\tx=%d\ty=%d\t"%(k,x,y))
                    zMin = np.min(cluster[k][:,2])
                    zMax = np.max(cluster[k][:,2])
                    changed += updateSurf(eMat[k][x][y], zMin, zMax)
        if info:
            print("Itteration %d, Changed %d"%(iterrations, changed))
        if changed == 0:
            break
    
    if info:
        for k in range(len(cluster)):
            surfTotalCount = []
            for x in range(ny):
                for y in range(nx):
                    numSurf = len(eMat[k][x][y].surf)
                    if numSurf >= len(surfTotalCount):
                        for i in range(numSurf-len(surfTotalCount)+1):
                            surfTotalCount.append(0)
                    surfTotalCount[numSurf]+=1
                    #print(surfPCD)
            if info:
                print("In cluster %d the elements per SPREAD surface count are as follows:"%k)
                print("numElement / numSurf: ",end="")
                for c in range(len(surfTotalCount)):
                    print("(%d/ %d)\t"%(surfTotalCount[c],c), end = "")
                print()
                    
        #print(surfPCD)
    if write:
        #start = clock_msg('*Exporting Everything',start,begining)
        
        filename = "../data/elements/surfPCD.pcd"
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(surfPCD)
        open3d.write_point_cloud(filename, pcd_export)
        filename = "../data/elements/surfPCD.pcd"
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(surfPCD)
        open3d.write_point_cloud(filename, pcd_export)  
        
        #print("After Merge")
        surfPCDmerged = createSurfPCD(nx,ny,eMat, cluster)
        filename = "../data/elements/surfPCD_merged.pcd"
        pcd_export = open3d.PointCloud()
        color = np.diag(np.divide([0, 255, 0],255))
        rgb = np.matmul(np.ones((len(surfPCDmerged),3)),color)
        pcd_export.colors = open3d.Vector3dVector(rgb)
        pcd_export.points = open3d.Vector3dVector(surfPCDmerged)
        open3d.write_point_cloud(filename, pcd_export) 
        
    
        
    start = clock_msg('*Combining surface points using clustering of normals',start,begining)
    surfPointsCombined = segmentNorms(eMat,cluster,nx,ny)
    
    
    if info:
        for k in range(len(cluster)):
            print("In cluster %d:"%k,end="\n\t")
            for layer in range(len(surfPointsCombined[k])):
                print("Surf %d: %d, "%(layer,len(surfPointsCombined[k][layer])), end="")
            print()
    
    '''
    for k in range(len(surfPointsCombined)):
        for layer in range(len(surfPointsCombined[k])):
            filename = "../data/elements/surfPCD_all_" + str(k) + "," + str(layer) + ".pcd"
            pcd_export = open3d.PointCloud()
            if layer == 0:
                color = np.diag(np.divide([255, 255, 255],255))
            else:
                color = np.diag(np.divide([0, 0, 0],255))
            rgb = np.matmul(np.ones((len(surfPointsCombined[k][layer]),3)),color)
            pcd_export.colors = open3d.Vector3dVector(rgb)
            res = np.array((surfPointsCombined[k][layer]))
            pcd_export.points = open3d.Vector3dVector(res)
            open3d.write_point_cloud(filename, pcd_export)
            
            
            res, eq = curveFit(surfPointsCombined[k][layer])
            filename = "../data/elements/surfPCD_Plane_" + str(k) + "," + str(layer) + ".pcd"
            pcd_export = open3d.PointCloud()
            color = np.diag(np.divide([255, 0, 255],255))
            rgb = np.matmul(np.ones((len(res),3)),color)
            pcd_export.colors = open3d.Vector3dVector(rgb)
            pcd_export.points = open3d.Vector3dVector(res)
            open3d.write_point_cloud(filename, pcd_export)
    '''
    start = clock_msg('*Subslice the elements into components',start,begining)
    #Next step is to use the surfaces to split the pier, piercap, and deck
    deck = []
    pierCap = []
    pier = []
    
    B = []
    #dz = zMax-zMin
    delta = 0.005
    numSurf = len(eMat[0][0][0].surf)
    pierArray = []
    pierCapArray = []
    deckArray = []
    #The subset of points that lie on the pier - Deck or pierCap - Deck interface
    rem_PD_Sub = [[] for i in range(len(cluster))]
    #The subset of points lying on the pier - pierCap interface
    rem_PC_Sub = [[] for i in range(len(cluster))]
    flag1 = False
    
    for k in range(len(cluster)):
        flag2 = False
        zBounds = []
        pierAreaSub[k] = pierAreaSub[k][np.argsort(pierAreaSub[k][:,2]), :]
        
        #For each layer, extract the min/max points of the planes
        #   Consider doing this to each element instead of each cluster if speed becomes an issue
        #   (Will cause less points to need to be compared to the planes)
        for layer in range(numSurf):
            _,eq = curveFit(surfPointsCombined[k][layer])
            res = findZBoundVals(eq,pierAreaSub[k])
            index = findZIndex(res, pierAreaSub[k])
            zBounds.append(index[0])
            zBounds.append(index[1])
        #For each layer, use the bounds found to quickly extract the slices
        #which correspond to pier/deck/ect.
        for layer in range(numSurf):
            if (not flag1):
                if numSurf == 1:
                    pier = pierAreaSub[k][:zBounds[0],:]
                    deck = pierAreaSub[k][zBounds[1]:,:]
    
                elif numSurf == 2:
                    pier = pierAreaSub[k][:zBounds[0],:]
                    pierCap = pierAreaSub[k][zBounds[1]:zBounds[2],:]
                    deck = pierAreaSub[k][zBounds[3]:,:]
    
                else:
                    print("*"*100)
                    print("Critical Error. Only 1 or 2 surfaces currently supported. Found " + str(len(numSurf)))
                    print("*"*100)
                flag1 = True
            else:
                if numSurf == 1:
                    pier = np.vstack((pier,pierAreaSub[k][:zBounds[0],:]))
                    deck = np.vstack((deck,pierAreaSub[k][zBounds[1]:,:]))
    
                elif numSurf == 2:
                    pier = np.vstack((pier,pierAreaSub[k][:zBounds[0],:]))
                    pierCap = np.vstack((pierCap,pierAreaSub[k][zBounds[1]:zBounds[2],:]))
                    deck = np.vstack((deck,pierAreaSub[k][zBounds[3]:,:]))
    
                    
                    
                    
            if (not flag2):
                if numSurf == 1:
                    rem_PD_Sub[k] = pierAreaSub[k][zBounds[0]:zBounds[1],:]
                elif numSurf == 2:
                    rem_PC_Sub[k] = pierAreaSub[k][zBounds[0]:zBounds[1],:]
                    rem_PD_Sub[k] = pierAreaSub[k][zBounds[2]:zBounds[3],:]
                flag2 = True
            else:
                if numSurf == 1:
                    rem_PD_Sub[k] = np.vstack((rem_PD_Sub[k],pierAreaSub[k][zBounds[0]:zBounds[1],:]))
                elif numSurf == 2:
                    rem_PC_Sub[k] = np.vstack((rem_PC_Sub[k],pierAreaSub[k][zBounds[0]:zBounds[1],:]))
                    rem_PD_Sub[k] = np.vstack((rem_PD_Sub[k],pierAreaSub[k][zBounds[2]:zBounds[3],:]))
                
        #For each layer, compare each point to it's corresponding planar value
        #and decide where to put it
            
        #want about 0.1 meter offset for the deck-pierCap inerface
        deltaTop = dz/200
        deltaBot = dz/200
        #Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        if numSurf == 1:
            C = curveFitQuad(surfPointsCombined[k][0])
            for point in rem_PD_Sub[k]:
                x = point[0]
                y = point[1]
                zfun = np.dot([1,x,y,x*y,x**2,y**2],C)
                if point[2]<zfun-deltaTop:
                    pierArray.append([point[0],point[1],point[2]])
                else:
                    deckArray.append([point[0],point[1],point[2]])
        if numSurf == 2:
            C0 = curveFitLinear(surfPointsCombined[k][0])
            C1 = curveFitQuad(surfPointsCombined[k][1])
            for point in rem_PD_Sub[k]:
                x = point[0]
                y = point[1]
                zfun = np.dot([1,x,y,x*y,x**2,y**2],C1)
                if point[2]<zfun-deltaTop:
                    pierCapArray.append([point[0],point[1],point[2]])
                else:
                    deckArray.append([point[0],point[1],point[2]])
                    
            for point in rem_PC_Sub[k]:
                x = point[0]
                y = point[1]
                #zfun = np.dot([1,x,y,x*y,x**2,y**2],C0)
                zfun = x*C0[0]+y*C0[1]+C0[2]
                if point[2]<zfun-deltaBot:
                    pierArray.append([point[0],point[1],point[2]])
                else:
                    pierCapArray.append([point[0],point[1],point[2]])
                    
                       
    pier = np.vstack((pier,np.array(pierArray)))
    pierCap = np.vstack((pierCap,np.array(pierCapArray)))
    deck = np.vstack((deck,np.array(deckArray)))
    
    
    #exportComponents2(np.array(deckArray),np.array(pierCapArray),np.array(pierArray))
            
            
    
    if write:
        exportComponents(deck,pierCap,pier)   
    
    
    #start = clock_msg('',start,begining)
    #return p
    return deck, pierCap, pier, start



'''
begining = time.perf_counter()
start = begining
write = True
pierArea = np.load("pierArea.npy")
'''
#deck, pierCap, pier, start = pierAreaSegmentation(pierArea,begining,start,write)

#p = pierAreaSegmentation(pierArea,begining,start,write)

