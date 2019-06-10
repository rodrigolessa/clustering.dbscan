# Indexing the dataset by quantifying each image in terms of shape.
# Apply the shape descriptor defined to every sprite in dataset.
# Frist we need the outline (or mask) of the object in the image 
# prior to applying Zernike moments. 
# In order to find the outline, we need to apply segmentation

# Import the necessary packages
from zernike_moments import ZernikeMoments
#from PIL import Image, ImageOps
import numpy as np
#import argparse
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
imageFolderConverted = imageFolder + '\\converted'
imageFolderThreshold = imageFolder + '\\threshold'
imageExtension = '.png'
imageFinder = '{}\\*{}'.format(imageFolderConverted, imageExtension)
imageDebug = 'butterfly-1'
imageMomentsFile = 'index.pkl'
imageRadius = 180
zernikeDegree = 16

index = {}

try:
	# If index file exists, try to delete
    os.remove(imageMomentsFile)
	# If folder to hold thresholder exists, try to delete
	#os.remove(imageFolderThreshold)
except OSError:
    pass

try:
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

startIndexing = time.time()

# Initialize descriptor with a radius of 160 pixels
zm = ZernikeMoments(imageRadius, zernikeDegree)

#print(imageFinder)

imagesInFolder = glob.glob(imageFinder)

qt = len(imagesInFolder)

print('images in the folder: {}'.format(qt))

i = 1

# Loop over the sprite images
for spritePath in imagesInFolder:
	# Extract image name, this will serve as unqiue key into the index dictionary.
	imageName = spritePath[spritePath.rfind('\\') + 1:].lower().replace(imageExtension, '')

	progress(i, qt)

	# then load the image.
	original = cv2.imread(spritePath)

	# Debugging: show original image
	if imageName.find(imageDebug) >= 0:
		cv2.imshow('original', original)
		cv2.waitKey(0)

	# Convert it to grayscale
	grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

	# Debugging: show original image
	if imageName.find(imageDebug) >= 0:
		cv2.imshow('grayscale', grayscale)
		cv2.waitKey(0)

	# Bilateral Filter can reduce unwanted noise very well
	blur = cv2.bilateralFilter(original, 9, 75, 75)

	# Debugging: pads along the 4 directions
	if imageName.find(imageDebug) >= 0:
		cv2.imshow('blur', blur)
		cv2.waitKey(0)

	# For segmentation: Flip the values of the pixels 
	# (black pixels are turned to white, and white pixels to black).
	thresh = cv2.bitwise_not(blur)

	# Debugging: Invert image
	if imageName.find(imageDebug) >= 0:
		cv2.imshow('inverted', thresh)
		cv2.waitKey(0)

	# Then, any pixel with a value greater than zero (black) is set to 255 (white)
	thresh[thresh > 0] = 255

	# Debugging: Invert image and threshold it
	if imageName.find(imageDebug) >= 0:
		cv2.imshow('thresholded', thresh)
		cv2.waitKey(0)

	# Accessing Image Properties
	# Image properties include number of rows, columns and channels, 
	# type of image data, number of pixels etc.
	# Shape of image is accessed by img.shape. It returns a tuple of number of rows, 
	# columns and channels (if image is color):
	outline = np.zeros(blur.shape, dtype = "uint8")

	# Initialize the outline image,
	# find the outermost contours (the outline) of the object, 
	# cv2.RETR_EXTERNAL - telling OpenCV to find only the outermost contours.
	# cv2.CHAIN_APPROX_SIMPLE - to compress and approximate the contours to save memory
	img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# Sort the contours based on their area, in descending order. 
	# keep only the largest contour and discard the others.
	contours = sorted(contours, key = cv2.contourArea, reverse = True)[0]

	# The outline is drawn as a filled in mask with white pixels:
	cv2.drawContours(outline, [contours], -1, 255, -1)

	# Debugging: just outline of the object
	if imageName.find(imageDebug) >= 0:
		cv2.imshow('outline', outline)
		cv2.waitKey(0)
	
	cv2.imwrite("{}\\{}.gif".format(imageFolderThreshold, imageName), outline)

	# Compute Zernike moments to characterize the shape of object outline
	moments = zm.describe(outline)

	# Debugging: analyse descriptions of form
	if imageName.find(imageDebug) >= 0:
		print(moments.shape)
		print('{}: {}'.format(imageName, moments))

	# then update the index
	index[imageName] = moments

	i+=1

cv2.destroyAllWindows()

# cPickle for writing the index in a file
with open(imageMomentsFile, "wb") as outputFile:
	cp.dump(index, outputFile, protocol=cp.HIGHEST_PROTOCOL)

doneIndexing = time.time()

elapsed = (doneIndexing - startIndexing) / 1000

print(" ")
print(elapsed)