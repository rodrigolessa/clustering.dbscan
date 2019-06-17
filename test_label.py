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

print('')
print('Names of the images by goups')
names = [t.split('\\')[3].split('-')[0] for t in imagesInFolder]
print(names[:30])

# Original labels
# Cria um array com uma sequencia numérica 
# atribuindo um valor para cada nome único nas imagens
labels_true = pd.factorize([k.split('-')[0] for k in sparse_matrix.keys()])[0]

print('')
print(labels_true[:30])