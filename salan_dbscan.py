# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:36:21 2019

@author: Alan
"""

import hdbscan
from matplotlib import pyplot as plt
import numpy as np


def hdbscan_fun(blobs, plot):
    numPoints = len(blobs)
    clusterer = hdbscan.HDBSCAN(min_samples=40,min_cluster_size=numPoints//100)
    clusterer.fit(blobs)
    
    labels_hdbscan = clusterer.labels_
    n_clusters_ = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)
    
    clusters = [blobs[labels_hdbscan == i] for i in range(n_clusters_)]
    
    if plot:
        unique_labels = set(labels_hdbscan)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]
        
            
            #plt.plot(clusters[k][:, 0], clusters[k][:, 1], 'o', markerfacecolor=tuple(col),
             #        markeredgecolor='k', markersize=10)
            plt.plot(blobs[labels_hdbscan == k][:, 0], blobs[labels_hdbscan == k][:, 1], 'o', markerfacecolor=tuple(col),
                     markeredgecolor='k', markersize=10)
    print("Number of clusters found: %d\t"%(n_clusters_))
    plt.show()    
    return clusters, labels_hdbscan    
    
    