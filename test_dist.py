from zernike_moments import ZernikeMoments
from scipy.spatial import distance as dist
import numpy as np
#import argparse
import cv2
import pickle as cp
import glob
import imutils
import os
import sys
import time

imageFolder = "D:\\mpeg7"
imageFolderConverted = '{}\\{}'.format(imageFolder, 'converted')
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'threshold')
imageExtension = '.png'
imageFinder = '{}\\*{}'.format(imageFolderConverted, imageExtension)
imageDebug = '{}{}'.format('butterfly-1', imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'

with open(imageMomentsFile, 'rb') as pickle_file:
    sparse_matrix = cp.load(pickle_file)

query = sparse_matrix['butterfly-1']

# initialize our dictionary of results
results = {}

# loop over the images in our index
for (k, features) in sparse_matrix.items():
    # Compute the distance between the query features
    # and features in our index, then update the results
    #d = dist.euclidean(query, features)
    d = dist.cosine(query, features)
    #d = np.linalg.norm(query - features)

    #print('Caracterídtica: {} - {}'.format(features[0], features[1]))

    #print('Distance: {}'.format(d))
    results[k] = d

# Sort our results, where a smaller distance indicates
# higher similarity
results = sorted([(v, k) for (k, v) in results.items()])[:20]

for r in results:
    #imageZeros = '{-:0>3}'.format(imageNumber)
    print("That object is: {}".format(r[1].upper()))