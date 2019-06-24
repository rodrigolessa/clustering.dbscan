from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets.samples_generator import make_blobs
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pickle as cp
import glob
import imutils
import os
import sys

imageFolder = "D:\\mpeg7"
imageFolderConverted = '{}\\{}'.format(imageFolder, 'converted')
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'threshold')
imageExtension = '.png'
imageFinder = '{}\\*{}'.format(imageFolderConverted, imageExtension)
imageDebug = '{}{}'.format('butterfly-1', imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'

# Read pickle file to a dict
#unpickled_df = pd.read_pickle(imageMomentsFile)
with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

# Original labels
labels_true = pd.factorize([k.split('-')[0] for k in sparse_matrix.keys()])[0]

#print(labels_true[:50])

# Convert the dict to a numpy array
X = np.array(list(sparse_matrix.values()))

# O data set de imagemMPEG7 possui 69 grupos
dbscan = DBSCAN(eps=0.01, metric='cosine', min_samples=3).fit(X)

#print(dbscan.labels_[:50])

# Return sequencial labels
labels = dbscan.labels_

# Gravar imagens nas pastas ou mostrar nomes agrupados

# pd = pandas

#adjusted_rand_score(labels, predict)
#adjusted_mutual_info_score(labels, predict)

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

n_noise_ = list(labels).count(-1)

print('')
print('Estimated number of clusters: %d' 
    % n_clusters_)
print('Estimated number of noise points: %d' 
    % n_noise_)
print("Homogeneity: %0.3f" 
    % metrics.homogeneity_score(labels_true, labels))
print("Completeness: %0.3f" 
    % metrics.completeness_score(labels_true, labels))
print("V-measure: %0.3f" 
    % metrics.v_measure_score(labels_true, labels))
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