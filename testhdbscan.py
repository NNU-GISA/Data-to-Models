# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:15:38 2019

@author: Alan
"""
import hdbscan
from sklearn.datasets import make_blobs
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


blobs, labels = make_blobs(n_samples=2000, n_features=2,random_state=0)

clusterer = hdbscan.HDBSCAN(min_samples=20,min_cluster_size=60)
clusterer.fit(blobs)

labels_hdbscan = clusterer.labels_
n_clusters_ = len(set(labels_hdbscan)) - (1 if -1 in labels_hdbscan else 0)

clusters = [blobs[labels_hdbscan == i] for i in range(n_clusters_)]

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
    
    
print("Number of actual clusters: %d\t Found: %d\t"%(len(set(labels)),n_clusters_))