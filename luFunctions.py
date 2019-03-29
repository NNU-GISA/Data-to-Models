# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:31:10 2019

@author: Alan
"""

import numpy as np
#import open3d#for pcd file io, and point normal calculation
from sklearn import decomposition#for pca
import time
#import bisect
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
    return xyz_oriented