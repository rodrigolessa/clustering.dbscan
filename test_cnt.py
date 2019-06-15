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

imageName = "test-11"
imagePath = 'D:\\mpeg7\\converted\\butterfly-1.png'
#imagePath = 'D:\\mpeg7\\converted\\apple-1.png'
#imagePath = 'D:\\mpeg7\\converted\\chicken-12.png'
#imagePath = 'D:\\mpeg7\\converted\\bird-11.png'
imageFolder = "D:\\mpeg7"
imageFolderConverted = imageFolder + '\\converted'
imageFolderThreshold = imageFolder + '\\threshold'
imageExtension = '.png'
imageFinder = '{}\\*{}'.format(imageFolderConverted, imageExtension)
imageMomentsFile = 'index.pkl'
imageRadius = 180
zernikeDegree = 16

index = {}

try:
	# If index file exists, try to delete
    os.remove(imageMomentsFile)
except OSError:
    pass

try:
    os.makedirs(imageFolderThreshold)
except OSError as e:
	pass

# Initialize descriptor with a radius of 160 pixels
zm = ZernikeMoments(imageRadius, zernikeDegree)

# then load the image.
original = cv2.imread(imagePath)

# Convert it to grayscale
grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

# Bilateral Filter can reduce unwanted noise very well
blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

# For segmentation: Flip the values of the pixels 
# (black pixels are turned to white, and white pixels to black).
#inverted = cv2.bitwise_not(blur)

# Then, any pixel with a value greater than zero (black) is set to 255 (white)
#thresh[thresh > 0] = 255
#_, threshold = cv2.threshold(inverted, 0 , 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
_, threshold = cv2.threshold(blur, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco

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
#img2, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
(contours, hierarchy) = cv2.findContours(threshold.copy(), 
	cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Sort the contours based on their area, in descending order. 
# TODO: keep only the largest contour and discard the others.
contours = sorted(contours, key = cv2.contourArea, reverse = True)[:2]

# The outline is drawn as a filled in mask with white pixels:
for cnt in contours:
	if(cv2.contourArea(cnt) > 0):
		cv2.drawContours(outline, [cnt], -1, 255, -1)

if (outline.any()):
	outline = cv2.bitwise_not(outline)

debug1 = np.hstack((grayscale, blur, threshold, outline))
cv2.imshow('grayscale + blur + threshold + outline', debug1)
cv2.waitKey(0)

# Find where the signature is and make a cropped region

y, x = outline.shape

# find where the black pixels are
points = np.argwhere(outline == 0) 
# store them in x,y coordinates instead of row, col indices
points = np.fliplr(points)

# create a rectangle around those points
x, y, w, h = cv2.boundingRect(points)

del points

# make the box a little bigger
x, y, w, h = x-10, y-10, w+20, h+20

if x < 0: x = 0
if y < 0: y = 0

signature = outline[y:y+h, x:x+w]

# Add border

size = 100

new = imutils.resize(signature, height=size)

if new.shape[1] > size:
	new = imutils.resize(new, width=size)

border_size_x = (size - new.shape[1])//2
border_size_y = (size - new.shape[0])//2

new = cv2.copyMakeBorder(
	new, 
	top=border_size_y + size, 
	bottom=border_size_y + size, 
	left=border_size_x + size, 
	right=border_size_x + size,
	borderType=cv2.BORDER_CONSTANT,
	value=[255,255,255]
	#cv2.BORDER_REPLICATE
)

cv2.imshow('new', new)
cv2.waitKey(0)
		
cv2.imwrite("{}\\{}.png".format(imageFolderThreshold, imageName), new)

# Compute Zernike moments to characterize the shape of object outline
moments = zm.describe(new)

print(moments.shape)
print('{}: {}'.format(imageName, moments))

cv2.destroyAllWindows()