from luFunctions import clock_msg,orientPCA,sortSliceX,assignXslice,removeDeckTop,sliceY,assignYslice, finalSegmentation, combineDeck, exportComponents

import numpy as np
import open3d#for pcd file io, and point normal calculation
#from sklearn import decomposition#for pca
import time
#import bisect
#from matplotlib import pyplot as plt#for histogram and 3d visualization
#from mpl_toolkits.mplot3d import Axes3D#for 3d visualization

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

write = False
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
pierArea, deckArea = assignYslice(ySlice, deckTop, p2, ny, zMax, zMin)

#Step 4: Segment pierArea into base components
start = clock_msg('Final Segmenting of Pier Areas (step4) using histograms of point normals',start,begining)
deck, pierCap, pier = finalSegmentation(pierArea,zMax,zMin,nb)

#Combine all elements back into deck array by going row by row
start = clock_msg('Combining All deck slices',start,begining)
deck = combineDeck(deck, deckX, deckArea, deckTop)

#Export the big 3: deck, pierCap, and pier
start = clock_msg('Exporting Deck,PierCap, and Pier Point Sets',start,begining)
exportComponents(deck,pierCap, pier)


start = clock_msg('',start,begining)

