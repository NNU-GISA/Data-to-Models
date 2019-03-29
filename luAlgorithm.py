import numpy as np
import open3d#for pcd file io, and point normal calculation
from sklearn import decomposition#for pca
import time
import bisect
from matplotlib import pyplot as plt#for histogram and 3d visualization
from mpl_toolkits.mplot3d import Axes3D#for 3d visualization

def showStats(mat):

    print("X axis values [min, mean, max]")
    print(str(round(np.min(mat[:,0]),2)) + "\t" + str(round(np.mean(mat[:,0]),2)) + "\t" + str(round(np.max(mat[:,0]),2)))
    print("\nY axis values [min, mean, max]")
    print(str(round(np.min(mat[:,1]),2)) + "\t" + str(round(np.mean(mat[:,1]),2)) + "\t" + str(round(np.max(mat[:,1]),2)))
    print("\nZ axis values [min, mean, max]")
    print(str(round(np.min(mat[:,2]),2)) + "\t" + str(round(np.mean(mat[:,2]),2)) + "\t" + str(round(np.max(mat[:,2]),2)))
    print()


#set manual factors
p1 = 0.25;
p2 = 0.30;
nx = 50;
ny = 10;
nb = 100;

start = time.clock()
print('\nLoading point cloud')
pcd_load = open3d.read_point_cloud("Bridge1ExtraClean.pcd")
xyz_load = np.asarray(pcd_load.points)
rgb_load = np.asarray(pcd_load.colors)




#Orient Bridge along x axis using PCA
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nOrienting point cloud along x axis')
start = time.clock()
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



'''
print("Before:\n")
showstats(xyz_load)
print("\nAfter:")
showstats(xyz_oriented)
'''

#slice x axis based on some delta (use 100 slices to start)
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nSorting')
start = time.clock()
xyz_sortedX_ascend = xyz_oriented[np.argsort(xyz_oriented[:,0]), :]
xyz = xyz_sortedX_ascend

xMin = np.min(xyz[:,0])
yMin = np.min(xyz[:,1])
zMin = np.min(xyz[:,2])
xMax = np.max(xyz[:,0])
yMax = np.max(xyz[:,1])
zMax = np.max(xyz[:,2])

delta = (1/nx)*(xMax-xMin)
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nSlicing along X')
start = time.clock()

BL = 0
step2 = []
for i in range(nx):
    BR = bisect.bisect_left(xyz[:,0],delta*(i+1)+xMin)
    step2.append(xyz[BL:BR,:])
    BL = BR

#step2
    #for each slice, check it against a user defined weighting value
    #and move the slice to step3 or step4
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nAssigning X slices (step2) as pier or deck areas')
start = time.clock()

red = np.diag(np.divide([255, 0, 0],255))
blue = np.diag(np.divide([0, 0, 255],255))
step3t = []
step3d = []
write = True
for i in range(len(step2)):
    filename = "step2/slice" + str(i) + ".pcd"
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
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nRemoving deck top from pier area X slices')
start = time.clock()
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
    
    filename = "step2_5/Dtop_" + str(len(deckTop)-1) + ".pcd"
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(deckTop[i])
    rgb = np.matmul(np.ones((len(deckTop[i]),3)),red)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud(filename, pcd_export)

    filename = "step2_5/Parea_" + str(i) + ".pcd"
    pcd_export = open3d.PointCloud()
    pcd_export.points = open3d.Vector3dVector(step3t[i])
    rgb = np.matmul(np.ones((len(step3t[i]),3)),blue)
    pcd_export.colors = open3d.Vector3dVector(rgb)
    open3d.write_point_cloud(filename, pcd_export)

    
#step3
    #for each pier slice, slice it again along the y axis
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nSlicing pier areas along Y axis')
start = time.clock()

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
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nAssigning Pier Areas (step3) as 20 pier or deck areas')
start = time.clock()

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
            filename = "step3/Pslice" + str(len(pierArea)-1) + ".pcd"
        else:
            rgb = np.matmul(np.ones((len(step3[k][i]),3)),red)
            deckArea.append(step3[k][i])
            filename = "step3/Dslice" + str(len(deckArea)-1) + ".pcd"
        if write:
            pcd_export.colors = open3d.Vector3dVector(rgb)
            open3d.write_point_cloud(filename, pcd_export)






#Step 4: Segment pierArea into base components
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nFinal Segmenting of Pier Areas (step4) using histograms of point normals')
start = time.clock()


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

print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nCombining All deck slices')
start = time.clock()

for i in range(len(step3d)):
    deck = np.vstack((deck,step3d[i]))

for i in range(len(deckArea)):
    deck = np.vstack((deck,deckArea[i]))

for i in range(len(deckTop)):
    deck = np.vstack((deck,deckTop[i]))
#consider a more efficient way of doing this. Doubling computational time is no bueno.
'''
sizeStep3d = 0
sizeDeckArea = 0
sizeDeckTop = 0

for i in range(len(step3d)):
    sizeStep3d += len(step3d[i])

for i in range(len(deckArea)):
    sizeDeckArea += len(deckArea[i])

for i in range(len(deckTop)):
    sizeDeckTop += len(deckTop[i])

deckSize = sizeStep3d + sizeDeckArea + sizeDeckTop + len(deck)
bigDeck = np.empty((deckSize,3))
pos = len(deck)
for i in range(len(step3d)):
    nex = len(step3d[i])
    deck[pos:pos+nex,:]=step3d[i]
    pos += nex

for i in range(len(deckArea)):
    nex = len(deckArea[i])
    deck[pos:pos+nex,:]=deckArea[i]
    pos += nex

for i in range(len(deckTop)):
    nex = len(deckTop[i])
    deck[pos:pos+nex,:]=deckTop[i]
    pos += nex
'''
print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))
print('\nExporting Deck,PierCap, and Pier Point Sets')
start = time.clock()

green = np.diag(np.divide([0, 255, 0],255))
pcd_export = open3d.PointCloud()
pcd_export.points = open3d.Vector3dVector(deck)
rgb = np.matmul(np.ones((len(deck),3)),red)
pcd_export.colors = open3d.Vector3dVector(rgb)
open3d.write_point_cloud("step4/deck.pcd", pcd_export)

pcd_export = open3d.PointCloud()
pcd_export.points = open3d.Vector3dVector(pierCap)
rgb = np.matmul(np.ones((len(pierCap),3)),green)
pcd_export.colors = open3d.Vector3dVector(rgb)
open3d.write_point_cloud("step4/pierCap.pcd", pcd_export)

pcd_export = open3d.PointCloud()
pcd_export.points = open3d.Vector3dVector(pier)
rgb = np.matmul(np.ones((len(pier),3)),blue)
pcd_export.colors = open3d.Vector3dVector(rgb)
open3d.write_point_cloud("step4/pier.pcd", pcd_export)





print('Delta: ' + str(time.clock()-start) + '\tTotal: ' + str(time.clock()))

