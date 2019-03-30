from luFunctions import clock_msg,orientPCA

import numpy as np
import open3d#for pcd file io, and point normal calculation
#from sklearn import decomposition#for pca
import time
import bisect
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
xyz_oriented, coef = orientPCA(xyz_load)

'''
print("Before:\n")
showstats(xyz_load)
print("\nAfter:")
showstats(xyz_oriented)
'''

#slice x axis based on some delta (use 100 slices to start)
start = clock_msg('Sorting',start,begining)
xyz_sortedX_ascend = xyz_oriented[np.argsort(xyz_oriented[:,0]), :]
xyz = xyz_sortedX_ascend

xMin = np.min(xyz[:,0])
yMin = np.min(xyz[:,1])
zMin = np.min(xyz[:,2])
xMax = np.max(xyz[:,0])
yMax = np.max(xyz[:,1])
zMax = np.max(xyz[:,2])

delta = (1/nx)*(xMax-xMin)
start = clock_msg('Slicing along X',start,begining)

BL = 0
step2 = []
for i in range(nx):
    BR = bisect.bisect_left(xyz[:,0],delta*(i+1)+xMin)
    step2.append(xyz[BL:BR,:])
    BL = BR

#step2
    #for each slice, check it against a user defined weighting value
    #and move the slice to step3 or step4
start = clock_msg('Assigning X slices (step2) as pier or deck areas',start,begining)

red = np.diag(np.divide([255, 0, 0],255))
blue = np.diag(np.divide([0, 0, 255],255))
step3t = []
step3d = []
write = True
for i in range(len(step2)):
    filename = "../data/step2/slice" + str(i) + ".pcd"
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(step2[i])
    localZmin = np.min(step2[i][:,2])
    localZmax = np.max(step2[i][:,2])
    
    if (localZmax-localZmin)>p1*(zMax-zMin):
        rgb = np.matmul(np.ones((len(step2[i]),3)),blue)
        step3t.append(step2[i])
    else:
        rgb = np.matmul(np.ones((len(step2[i]),3)),red)
        step3d.append(step2[i])
    if write:
        pcd_export.colors = open3d.Vector3dVector(rgb)
        open3d.write_point_cloud(filename, pcd_export)
#step2.5
    #for each pier slice, remove the deck top and set it aside
start = clock_msg('Removing deck top from pier area X slices',start,begining)

deckTop = []

for i in range(len(step3t)):
    #print("Length slice before: " + str(len(step3t[i])))
    yMin = np.min(step3t[i][:,1])
    yMax = np.max(step3t[i][:,1])
    zMax = np.max(step3t[i][:,2])
    deltaY = (yMax-yMin)
    deltaZ = 0.05*deltaY
    step3t[i] = step3t[i][np.argsort(step3t[i][:,2]), :]
    BR = bisect.bisect_left(step3t[i][:,2],zMax-deltaZ)
    deckTop.append(step3t[i][BR:,:])
    step3t[i] = step3t[i][:BR,:]
    #print("yMin %.3f\t yMax %.3f\t deltaY %.3f\t deltaZ %.3f\t BR %.3f\t"%(yMin,yMax,deltaY,deltaZ,BR))
    #print("Length slice after: " + str(len(step3t[i])))
    
    filename = "../data/step2_5/Dtop_" + str(len(deckTop)-1) + ".pcd"
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(deckTop[i])
    rgb = np.matmul(np.ones((len(deckTop[i]),3)),red)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud(filename, pcd_export)

    filename = "../data/step2_5/Parea_" + str(i) + ".pcd"
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(step3t[i])
    rgb = np.matmul(np.ones((len(step3t[i]),3)),blue)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud(filename, pcd_export)

    
#step3
    #for each pier slice, slice it again along the y axis
start = clock_msg('Slicing pier areas along Y axis',start,begining)

step3 = []
step3_hold = []

for i in range(len(step3t)):
    slic = step3t[i]
    BL=0
    yMin = np.min(slic[:,1])
    yMax = np.max(slic[:,1])
    delta = (1/ny)*(yMax-yMin)
    slic = slic[np.argsort(slic[:,1]), :]
    for i in range(20):
        BR = bisect.bisect_left(slic[:,1],delta*(i+1)+yMin)
        step3_hold.append(slic[BL:BR,:])
        BL = BR
    
    step3.append(step3_hold)
    step3_hold = []

#assign by user value
start = clock_msg('Assigning Pier Areas (step3) as 20 pier or deck areas',start,begining)

pierArea = []
deckArea = []
#don't forget step3d contains all the x slices of the deck
write = True
#deck top is in deckTop[i][row,3] and should correspond to step3[i][:][row,3]
for k in range(len(step3)):#42 x slices
    #note, using the whole x-slice zmax as the local zmax is not##################################################
    #entirely accurate. If issues arise, consider finding
    #the local zmax for each box seperately
    localZmax = np.max(deckTop[k][:,2])
    for i in range(ny):#20 y slices per x slice
        
        pcd_export = open3d.PointCloud()
        pcd_export.points = open3d.Vector3dVector(step3[k][i])
        localZmin = np.min(step3[k][i][:,2])
        #localZmax = np.max(step3[k][i][:,2])
        
        if (localZmax-localZmin)>p2*(zMax-zMin):
            rgb = np.matmul(np.ones((len(step3[k][i]),3)),blue)
            pierArea.append(step3[k][i])
            filename = "../data/step3/Pslice" + str(len(pierArea)-1) + ".pcd"
        else:
            rgb = np.matmul(np.ones((len(step3[k][i]),3)),red)
            deckArea.append(step3[k][i])
            filename = "../data/step3/Dslice" + str(len(deckArea)-1) + ".pcd"
        if write:
            pcd_export.colors = open3d.Vector3dVector(rgb)
            open3d.write_point_cloud(filename, pcd_export)






#Step 4: Segment pierArea into base components
start = clock_msg('Final Segmenting of Pier Areas (step4) using histograms of point normals',start,begining)

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

start = clock_msg('Combining All deck slices',start,begining)

#big_deck = []
'''
for i in range(len(step3d)):
    deck = np.vstack((deck,step3d[i]))
    #big_deck.append(step3d[i])

for i in range(len(deckArea)):
    deck = np.vstack((deck,deckArea[i]))
    #big_deck.append(deckArea[i])

for i in range(len(deckTop)):
    deck = np.vstack((deck,deckTop[i]))
    #big_deck.append(deckTop[i])
    
   ''' 

#consider a more efficient way of doing this. Doubling computational time is no bueno.

sizeStep3d = 0
sizeDeckArea = 0
sizeDeckTop = 0
#try looping over each row in each element of each list and add them row by row to deck
for i in range(len(step3d)):
    sizeStep3d += len(step3d[i])

for i in range(len(deckArea)):
    sizeDeckArea += len(deckArea[i])

for i in range(len(deckTop)):
    sizeDeckTop += len(deckTop[i])

deckSize = sizeStep3d + sizeDeckArea + sizeDeckTop + len(deck)
bigDeck = np.empty((deckSize,3))
pos = 0
#transfer deck to bigDeck
nex = len(deck)
for k in range(len(deck)):
    bigDeck[pos+k,:]=deck[k,:]

#deck[pos:pos+nex,:]=step3d[i]    
pos += nex
for i in range(len(step3d)):
    nex = len(step3d[i])
    for k in range(len(step3d[i])):
        bigDeck[pos+k,:]=step3d[i][k,:]
    #deck[pos:pos+nex,:]=step3d[i]    
    pos += nex
for i in range(len(deckArea)):
    nex = len(deckArea[i])
    for k in range(len(deckArea[i])):
        bigDeck[pos+k,:]=deckArea[i][k,:]
    #deck[pos:pos+nex,:]=step3d[i]    
    pos += nex
for i in range(len(deckTop)):
    nex = len(deckTop[i])
    for k in range(len(deckTop[i])):
        bigDeck[pos+k,:]=deckTop[i][k,:]
    #deck[pos:pos+nex,:]=step3d[i]    
    pos += nex

start = clock_msg('Exporting Deck,PierCap, and Pier Point Sets',start,begining)
deck = bigDeck
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


start = clock_msg('',start,begining)

