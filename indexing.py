# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

# Import the necessary packages
from zernike_moments import ZernikeMoments
#from PIL import Image, ImageOps
from my_pre_processing import Mpeg7PreProcessing
import numpy as np
import pandas as pd
import cv2
import pickle as cp
import glob
import imutils
import os
import sys
import time

# Construct the argument parser and parse the arguments
#ap = argparse.ArgumentParser()
#ap.add_argument("-f", "--folder", required = True, help = "Path to where the files has stored")
#ap.add_argument("-e", "--extension", required = True, help = "Extension of the images")
#ap.add_argument("-i", "--index", required = True, help = "Path to where the index file will be stored")

#args = vars(ap.parse_args())

imageFolder = "D:\\mpeg7"
imageFolderConverted = '{}\\{}'.format(imageFolder, 'converted')
imageFolderThreshold = '{}\\{}'.format(imageFolder, 'threshold')
imageExtension = '.png'
imageFinder = '{}\\*{}'.format(imageFolderConverted, imageExtension)
imageDebug = '{}{}'.format('butterfly-1', imageExtension)
imagesInFolder = glob.glob(imageFinder)
imageMomentsFile = 'index.pkl'
imageSize = 180
imageRadius = 180
zernikeDegree = 16

# initialize our dictionary
index = {}

qtd = len(imagesInFolder)

i = 1

#print(imageFinder)
#print('images in the folder: {}'.format(qt))

try:
	# If index file exists, try to delete
    os.remove(imageMomentsFile)
except OSError:
    pass

try:
	# If folder to hold thresholder exists, try to delete
    os.makedirs(imageFolderThreshold)
except OSError as e:
	pass
	#import errno
    #if e.errno != errno.EEXIST:
    #raise

# Simulate a progress bar
def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s ...%s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

# Initialize descriptor with a radius of 160 pixels
zm = ZernikeMoments(imageRadius, zernikeDegree)
m7 = Mpeg7PreProcessing(imageDebug, imageSize, False)

startIndexing = time.time()

# Loop over the sprite images
for spritePath in imagesInFolder:
	
	# Extract image name, this will serve as unqiue key into the index dictionary.
	imageName = spritePath[spritePath.rfind('\\') + 1:].lower().replace(imageExtension, '')

	progress(i, qtd)
	
	outline = m7.getTheBestContour(spritePath)

	# TODO: Draw center of mass
	cv2.imwrite("{}\\{}.png".format(imageFolderThreshold, imageName), outline)

	# Compute Zernike moments to characterize the shape of object outline
	moments = zm.describe(outline)

	# Debugging: analyse descriptions of form
	#if imageName.find(imageDebug) >= 0:
		#print(moments.shape)
		#print('{}: {}'.format(imageName, moments))

	# then update the index
	index[imageName] = moments

	i+=1

#cv2.destroyAllWindows()

# cPickle for writing the index in a file
with open(imageMomentsFile, "wb") as outputFile:
	cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)

doneIndexing = time.time()

elapsed = (doneIndexing - startIndexing) / 1000

print(" ")
print(elapsed)
