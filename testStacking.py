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