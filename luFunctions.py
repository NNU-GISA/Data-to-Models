# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:31:10 2019

@author: Alan
"""

import numpy as np
import open3d#for pcd file io, and point normal calculation
from sklearn import decomposition#for pca
import time
import bisect
#from matplotlib import pyplot as plt#for histogram and 3d visualization
#from mpl_toolkits.mplot3d import Axes3D#for 3d visualization

def showStats(mat):

    print("X axis values [min, mean, max]")
    print(str(round(np.min(mat[:,0]),2)) + "\t" + str(round(np.mean(mat[:,0]),2)) + "\t" + str(round(np.max(mat[:,0]),2)))
    print("\nY axis values [min, mean, max]")
    print(str(round(np.min(mat[:,1]),2)) + "\t" + str(round(np.mean(mat[:,1]),2)) + "\t" + str(round(np.max(mat[:,1]),2)))
    print("\nZ axis values [min, mean, max]")
    print(str(round(np.min(mat[:,2]),2)) + "\t" + str(round(np.mean(mat[:,2]),2)) + "\t" + str(round(np.max(mat[:,2]),2)))
    print()
    
def clock_msg(msg,start,begining):
    print('Delta: ' + str(round(time.perf_counter()-start,5)) + '\tTotal: ' + str(round(time.perf_counter()-begining,5)))
    print('\n' + msg)
    start = time.perf_counter()
    return start

def orientPCA(xyz_load):
    
    pca = decomposition.PCA(n_components=2)
    x = xyz_load[:,0:2]
    pca.fit(x)
    mat = pca.components_
    coef = np.zeros((3,3))
    coef[2,2] = 1
    
    for i in range(2):
        for j in range(2):
            coef[i,j]=mat[i,j]
    
    
    coef[0,1] = -coef[0,1]
    coef[1,0] = -coef[1,0]
    xyz_oriented = np.matmul(xyz_load,coef)
    return xyz_oriented, coef


def sortSliceX(xyz_oriented, xMin, xMax, nx):
    xyz_sortedX_ascend = xyz_oriented[np.argsort(xyz_oriented[:,0]), :]
    xyz = xyz_sortedX_ascend
    
    delta = (1/nx)*(xMax-xMin)
    BL = 0
    xSlice = []
    for i in range(nx):
        BR = bisect.bisect_left(xyz[:,0],delta*(i+1)+xMin)
        xSlice.append(xyz[BL:BR,:])
        BL = BR
    return xSlice, xyz

def assignXslice(xSlice, p1, zMax, zMin, write):
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    notDeckX = []
    deckX = []
    
    for i in range(len(xSlice)):
        filename = "../data/step2/slice" + str(i) + ".pcd"
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(xSlice[i])
        localZmin = np.min(xSlice[i][:,2])
        localZmax = np.max(xSlice[i][:,2])
        
        if (localZmax-localZmin)>p1*(zMax-zMin):
            rgb = np.matmul(np.ones((len(xSlice[i]),3)),blue)
            notDeckX.append(xSlice[i])
        else:
            rgb = np.matmul(np.ones((len(xSlice[i]),3)),red)
            deckX.append(xSlice[i])
        if write:
            pcd_export.colors = open3d.Vector3dVector(rgb)
            open3d.write_point_cloud(filename, pcd_export)
    return deckX, notDeckX

def removeDeckTop(notDeckX, write):
    deckTop = []
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    for i in range(len(notDeckX)):
        #print("Length slice before: " + str(len(notDeckX[i])))
        yMin = np.min(notDeckX[i][:,1])
        yMax = np.max(notDeckX[i][:,1])
        zMax = np.max(notDeckX[i][:,2])
        deltaY = (yMax-yMin)
        deltaZ = 0.05*deltaY
        notDeckX[i] = notDeckX[i][np.argsort(notDeckX[i][:,2]), :]
        BR = bisect.bisect_left(notDeckX[i][:,2],zMax-deltaZ)
        deckTop.append(notDeckX[i][BR:,:])
        notDeckX[i] = notDeckX[i][:BR,:]
        #print("yMin %.3f\t yMax %.3f\t deltaY %.3f\t deltaZ %.3f\t BR %.3f\t"%(yMin,yMax,deltaY,deltaZ,BR))
        #print("Length slice after: " + str(len(notDeckX[i])))
        if write:
            filename = "../data/step2_5/Dtop_" + str(len(deckTop)-1) + ".pcd"
            pcd_export = open3d.PointCloud()
            pcd_export.points = open3d.Vector3dVector(deckTop[i])
            rgb = np.matmul(np.ones((len(deckTop[i]),3)),red)
            pcd_export.colors = open3d.Vector3dVector(rgb)
            open3d.write_point_cloud(filename, pcd_export)
        
            filename = "../data/step2_5/Parea_" + str(i) + ".pcd"
            pcd_export = open3d.PointCloud()
            pcd_export.points = open3d.Vector3dVector(notDeckX[i])
            rgb = np.matmul(np.ones((len(notDeckX[i]),3)),blue)
            pcd_export.colors = open3d.Vector3dVector(rgb)
            open3d.write_point_cloud(filename, pcd_export)
    return notDeckX, deckTop


def sliceY(notDeckX,ny):
    ySlice = []
    ySlice_hold = []
    
    for i in range(len(notDeckX)):
        slic = notDeckX[i]
        BL=0
        yMin = np.min(slic[:,1])
        yMax = np.max(slic[:,1])
        delta = (1/ny)*(yMax-yMin)
        slic = slic[np.argsort(slic[:,1]), :]
        for i in range(20):
            BR = bisect.bisect_left(slic[:,1],delta*(i+1)+yMin)
            ySlice_hold.append(slic[BL:BR,:])
            BL = BR
        
        ySlice.append(ySlice_hold)
        ySlice_hold = []
    return ySlice


def assignYslice(ySlice, deckTop, p2, ny, zMax, zMin, write):
    pierArea = []
    deckArea = []
    #don't forget deckX contains all the x slices of the deck
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    #deck top is in deckTop[i][row,3] and should correspond to ySlice[i][:][row,3]
    for k in range(len(ySlice)):#42 x slices
        #note, using the whole x-slice zmax as the local zmax is not##################################################
        #entirely accurate. If issues arise, consider finding
        #the local zmax for each box seperately
        localZmax = np.max(deckTop[k][:,2])
        for i in range(ny):#20 y slices per x slice
            
            pcd_export = open3d.PointCloud()
            pcd_export.points = open3d.Vector3dVector(ySlice[k][i])
            localZmin = np.min(ySlice[k][i][:,2])
            #localZmax = np.max(ySlice[k][i][:,2])
            
            if (localZmax-localZmin)>p2*(zMax-zMin):
                rgb = np.matmul(np.ones((len(ySlice[k][i]),3)),blue)
                pierArea.append(ySlice[k][i])
                filename = "../data/step3/nDslice_" + str(len(pierArea)-1) + ".pcd"
            else:
                rgb = np.matmul(np.ones((len(ySlice[k][i]),3)),red)
                deckArea.append(ySlice[k][i])
                filename = "../data/step3/Dslice" + str(len(deckArea)-1) + ".pcd"
            if write:
                pcd_export.colors = open3d.Vector3dVector(rgb)
                open3d.write_point_cloud(filename, pcd_export)
                
    return pierArea, deckArea



def finalSegmentation(pierArea,zMax,zMin,nb):
    for i in range(len(pierArea)):
        #print("i=%d"%i)
        #convert numpy array to point cloud object
        pcd = open3d.PointCloud()
        index = pierArea[i][:,2]>(0.5*(zMax-zMin)+zMin)
        PAsub = pierArea[i][index,:]
        pcd.points = open3d.Vector3dVector(PAsub)
        #estimate normals for every point
        open3d.estimate_normals(pcd, search_param = open3d.KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
        n = np.asarray(pcd.normals)
        #filter to just vertical normals, create a histogram of these
        ind = abs(n[:,2]) > 0.99
        hist = np.histogram(PAsub[:,2],range=((0.5*(zMax-zMin)+zMin),zMax), bins=nb)
        hist = np.histogram(PAsub[ind,2],range=((0.5*(zMax-zMin)+zMin),zMax), bins=nb)
        #hist = plt.hist(PAsub[:,2],range=((0.5*(zMax-zMin)+zMin),zMax), bins=nb)
        #hist = plt.hist(PAsub[ind,2],range=((0.5*(zMax-zMin)+zMin),zMax), bins=nb)
        #plt.show()
    
        ind = hist[0]>300*(20/nb)
        pos = np.where(ind)[0]
        #Merge adjacent non-zero bins (where nonzero is defined to be >100)
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
            
    
        #using the histograms, calculate the boundary values for each
        PA = pierArea[i]
        PA = PA[np.argsort(PA[:,2]), :]
        if len(pos)==2:
            p1 = pos[0]#p1 is the lower boundary position in the histogram index
            p2 = pos[1]#p2 is the upper boundary
            b1_left = hist[1][p1]
    
            b2_left = hist[1][p2]#the lower z VALUE @ the deck/pier cap boundary
    
            b2_leftIndex = bisect.bisect_left(PA[:,2],b2_left)
            b1_leftIndex = bisect.bisect_left(PA[:,2],b1_left)
            
    
        #add in nearest neighbor theifing here    
        elif len(pos)==1:
            p2 = pos[0]
            b2_left = hist[1][p2]
            
            b2_leftIndex = bisect.bisect_left(PA[:,2],b2_left)
    
            b1_left = np.min(PAsub[:,2])
            b1_leftIndex = bisect.bisect_left(PA[:,2],b1_left)
            
            
        #deck is all points including and above the upper boundary (b2)
        #pier cap is all points including and above the lower boundary (b1) after deck is removed
        #pier is whatever is left (below and excluding b1)
        if i==0:
            deck = PA[b2_leftIndex:,:]
            pierCap = PA[b1_leftIndex:b2_leftIndex,:]
            pier = PA[:b1_leftIndex,:]
        else:
            
            deck = np.vstack((deck,(PA[b2_leftIndex:,:])))
            pierCap = np.vstack((pierCap,(PA[b1_leftIndex:b2_leftIndex,:])))
            pier = np.vstack((pier,(PA[:b1_leftIndex,:])))
    return deck, pierCap, pier


def combineDeck(deck, deckX, deckArea, deckTop):
    sizedeckX = 0
    sizeDeckArea = 0
    sizeDeckTop = 0
    #try looping over each row in each element of each list and add them row by row to deck
    for i in range(len(deckX)):
        sizedeckX += len(deckX[i])
    
    for i in range(len(deckArea)):
        sizeDeckArea += len(deckArea[i])
    
    for i in range(len(deckTop)):
        sizeDeckTop += len(deckTop[i])
    
    deckSize = sizedeckX + sizeDeckArea + sizeDeckTop + len(deck)
    bigDeck = np.empty((deckSize,3))
    pos = 0
    #transfer deck to bigDeck
    nex = len(deck)
    for k in range(len(deck)):
        bigDeck[pos+k,:]=deck[k,:]
    
    #deck[pos:pos+nex,:]=deckX[i]    
    pos += nex
    for i in range(len(deckX)):
        nex = len(deckX[i])
        for k in range(len(deckX[i])):
            bigDeck[pos+k,:]=deckX[i][k,:]
        #deck[pos:pos+nex,:]=deckX[i]    
        pos += nex
    for i in range(len(deckArea)):
        nex = len(deckArea[i])
        for k in range(len(deckArea[i])):
            bigDeck[pos+k,:]=deckArea[i][k,:]
        #deck[pos:pos+nex,:]=deckX[i]    
        pos += nex
    for i in range(len(deckTop)):
        nex = len(deckTop[i])
        for k in range(len(deckTop[i])):
            bigDeck[pos+k,:]=deckTop[i][k,:]
        #deck[pos:pos+nex,:]=deckX[i]    
        pos += nex

    return bigDeck

def exportComponents(deck,pierCap, pier):
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    green = np.diag(np.divide([0, 255, 0],255))
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(deck)
    rgb = np.matmul(np.ones((len(deck),3)),red)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/step4/deck.pcd", pcd_export)
    
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(pierCap)
    rgb = np.matmul(np.ones((len(pierCap),3)),green)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/step4/pierCap.pcd", pcd_export)
    
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(pier)
    rgb = np.matmul(np.ones((len(pier),3)),blue)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud("../data/step4/pier.pcd", pcd_export)
    return 1


def exportComponentList(data,name,color):
    
    if type(data)==type([]):
        for i in range(len(data)):
            #print("len(data)=%d\ti=%d"%(len(data),i))
            filename = "../data/cluster/" + name + "_" + str(i) + "_.pcd"
            #print(filename)
            pcd_export = open3d.PointCloud()
            pcd_export.points = open3d.Vector3dVector(data[i])
            rgb = np.matmul(np.ones((len(data[i]),3)),color)
            pcd_export.colors = open3d.Vector3dVector(rgb)
            #print(pcd_export)
            open3d.write_point_cloud(filename, pcd_export)
            
    else:
        filename = "../data/cluster/" + name + ".pcd"
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(data)
        rgb = np.matmul(np.ones((len(data),3)),color)
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud(filename, pcd_export)
                
        


