from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pickle as cp
import glob
import imutils
import os
import sys

# Check the Matplotlib Version 
print ("Matplotlib Version", matplotlib.__version__)

# #############################################################################
# Generate sample data

centers = [[1, 1], [-1, -1], [1, -1]]

# The true cluster
X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

print('')
print('Exemplo de uma matriz de entrada para o DBSCAN, coordenadas em 2D:')
print(X[:5])
print('')
print('5 primeiros Labels anotados/conhecidos da matriz:')
print(labels_true[:5])
print('')
print('Elemento da posição 100:')
print(X[99])
print('')
print('{} - Features em cada elemento'.format(len(X[0])))
print('')
print('{} - Total de elementos na matriz'.format(len(X)))

# The idea behind StandardScaler is that it will transform your data 
# such that its distribution will have a mean value 0 and standard deviation of 1. 
# Given the distribution of the data, 
# each value in the dataset will have the sample mean value subtracted, 
# and then divided by the standard deviation of the whole dataset.
scaled = StandardScaler().fit_transform(X)

print('')
print('Exemplo da matriz redistribuida por um valor médio:')
print(scaled[:5])

print('')
print('Elemento da posição 100:')
print(scaled[99])

# #############################################################################
# Compute DBSCAN

db = DBSCAN(algorithm='auto', eps=0.3, metric='cosine', min_samples=8).fit(scaled)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

# The predicted cluster
labels = db.labels_

print('')
print('5 primeiros Labels agrupados pelo DBSCAN:')
print ('{}, do tipo: {}'.format(labels[:5], labels.dtype))

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)

print('')
print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# The Rand Index computes a similarity measure between two clusterings by considering 
# all pairs of samples and counting pairs that are assigned in the same 
# or different clusters in the predicted and true clusterings.
print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels, average_method='arithmetic'))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))

# #############################################################################
# Plot result

# Black removed and is used for noise instead.
unique_labels = set(labels)

colors = [plt.cm.Spectral(each)
          for each in np.linspace(0, 1, len(unique_labels))]

for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = [0, 0, 0, 1]

    class_member_mask = (labels == k)

    xy = X[class_member_mask & core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=14)

    xy = X[class_member_mask & ~core_samples_mask]
    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
             markeredgecolor='k', markersize=6)

plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()