﻿from sklearn import metrics
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

# Convert the dict to a numpy array
X = np.array(list(sparse_matrix.values()))

# O data set de imagemMPEG7 possui 69 grupos
dbscan = DBSCAN(algorithm='auto', eps=0.02, metric='cosine', min_samples=4).fit(X)

# Return sequencial labels
labels = dbscan.labels_

# pd = pandas

#adjusted_rand_score(labels, predict)
#adjusted_mutual_info_score(labels, predict)

# #############################################################################
# Plot result