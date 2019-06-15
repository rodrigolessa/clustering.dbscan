# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

from zernike_moments import ZernikeMoments
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScale
import numpy as np
import cv2
#import argparse
#import matplotlib.pyplot as plt
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

sparse_matrix = cp.load(imageMomentsFile)
 
# Tests
#X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
#clustering = DBSCAN(eps=3, min_samples=2).fit(X)
#print(clustering.labels_)
#print(clustering)

# loop over the images in our index
for (k, features) in sparse_matrix.items():
    print(k)
    print(features)
    # Compute the distance between the query features
    # and features in our index, then update the results
    #d = dist.euclidean(queryFeatures, features)

X = StandardScaler().fit_transform(sparse_matrix)

dbscan = DBSCAN(algorithm='auto', eps=0.03, metric='cosine', min_samples=3).fit(X)
#MPEG7dataset.zip
print(dbscan.labels_)

# Original labels
labels = pd.factorize([t.split('-')[0] for t in imagesInFolder])[0]

print(labels)

#Dá as labels em números e não texto. pd = pandas
#predict = dbscan.labels_
#adjusted_rand_score(labels, predict)
#adjusted_mutual_info_score(labels, predict)