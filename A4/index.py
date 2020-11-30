# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:44:46 2019

@author: user
"""

# import the necessary packages
from ColorDescriptor import ColorDescriptor
import glob2 as glob
import cv2
 
 
# initialize the color descriptor
cd = ColorDescriptor((8, 12, 3))

output = open('index.csv', "w")
 
# use glob to grab the image paths and loop over them
for imagePath in glob.glob("clean_set" + "/*.jpg"):
	# extract the image ID (i.e. the unique filename) from the image
	# path and load the image itself
	imageID = imagePath[imagePath.rfind("/") + 1:]
	image = cv2.imread(imagePath)
 
	# describe the image
	features = cd.describe(image)
 
	# write the features to file
	features = [str(f) for f in features]
	output.write("%s,%s\n" % (imageID, ",".join(features)))
 
# close the index file
output.close()

