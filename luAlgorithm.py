from luFunctions import clock_msg,orientPCA,sortSliceX,assignXslice,removeDeckTop,sliceY,assignYslice, finalSegmentation, combineDeck, exportComponents, exportComponentList
import salan_dbscan
import numpy as np
import open3d#for pcd file io, and point normal calculation
#from sklearn import decomposition#for pca
import time
#import bisect
#from matplotlib import pyplot as plt#for histogram and 3d visualization
#from mpl_toolkits.mplot3d import Axes3D#for 3d visualization
import pandas as pd
import hdbscan

#set manual factors
p1 = 0.25;
p2 = 0.30;
nx = 50;
ny = 10;
nb = 100;

begining = time.perf_counter()
start = begining
print('\nLoading point cloud')
pcd_load = open3d.read_point_cloud("..\data\Bridge1ExtraClean.pcd")
xyz_load = np.asarray(pcd_load.points)
rgb_load = np.asarray(pcd_load.colors)

#Orient Bridge along x axis using PCA
start = clock_msg('Orienting point cloud along x axis',start,begining)
xyz, coef = orientPCA(xyz_load)

xMin = np.min(xyz[:,0])
yMin = np.min(xyz[:,1])
zMin = np.min(xyz[:,2])
xMax = np.max(xyz[:,0])
yMax = np.max(xyz[:,1])
zMax = np.max(xyz[:,2])
#slice x axis based on some delta (use 100 slices to start)
start = clock_msg('Sort and Slice X',start,begining)
xSlice, xyz = sortSliceX(xyz, xMin, xMax, nx)


#step2
    #for each slice, check it against a user defined weighting value
    #and move the slice to step3 or step4
start = clock_msg('Assigning X slices (step2) as pier or deck areas',start,begining)

write = True
deckX, notDeckX = assignXslice(xSlice, p1, zMax, zMin, write)
#step2.5
    #for each pier slice, remove the deck top and set it aside
start = clock_msg('Removing deck top from pier area X slices',start,begining)
notDeckX, deckTop = removeDeckTop(notDeckX, write)

    
#step3
    #for each pier slice, slice it again along the y axis
start = clock_msg('Slicing pier areas along Y axis',start,begining)
ySlice = sliceY(notDeckX,ny)


#assign by user value
start = clock_msg('Assigning Pier Areas (step3) as 20 pier or deck areas',start,begining)
pierArea, deckArea = assignYslice(ySlice, deckTop, p2, ny, zMax, zMin, write)

#Step 4: Segment pierArea into base components
start = clock_msg('Final Segmenting of Pier Areas (step4) using histograms of point normals',start,begining)
deck, pierCap, pier = finalSegmentation(pierArea,zMax,zMin,nb)

#Combine all elements back into deck array by going row by row
start = clock_msg('Combining All deck slices',start,begining)
deck = combineDeck(deck, deckX, deckArea, deckTop)
exportComponents(deck,pierCap,pier)

clusterFlag = 0
if clusterFlag:
    #attempt to cluster the component point clouds (pier used for testing)
    start = clock_msg('Clustering Piers',start,begining)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pier)
    pcd = open3d.voxel_down_sample(pcd, voxel_size = 0.1)
    xyz= np.asarray(pcd.points)
    clusterPier, labelsPier = salan_dbscan.hdbscan_fun(xyz,True)
    
    start = clock_msg('Clustering PierCaps',start,begining)
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(pierCap)
    pcd = open3d.voxel_down_sample(pcd, voxel_size = 0.1)
    xyz= np.asarray(pcd.points)
    clusterPierCap, labelsPierCap = salan_dbscan.hdbscan_fun(xyz,True)
    
    #using the clustering data, find the bounding boxes for each 
    #cluster and extract those points from the FULL point set
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
    
    scale = 1
    bounds = np.zeros((len(clusterPier),4))
    for k in range(len(clusterPier)):
        bounds[k] = xyBounds(clusterPier[k], scale)
    cpFull = extract(pier, bounds)
    
    #cluster the pierCap next
    scale = 1
    bounds = np.zeros((len(clusterPierCap),4))
    for k in range(len(clusterPierCap)):
        bounds[k] = xyBounds(clusterPierCap[k], scale)
    cpcFull = extract(pierCap, bounds)
    
    #Export the big 3: deck, pierCap, and pier
    start = clock_msg('Exporting Deck,PierCap, and Pier Point Sets',start,begining)
    #exportComponents(deck,pierCap, pier)
    red = np.diag(np.divide([255, 0, 0],255))
    blue = np.diag(np.divide([0, 0, 255],255))
    green = np.diag(np.divide([0, 255, 0],255))
    exportComponentList(cpFull,"clusterPier",blue)
    exportComponentList(cpcFull,"clusterPierCap",green)
    exportComponentList(deck,"deck",red)



start = clock_msg('',start,begining)

