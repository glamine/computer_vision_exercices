# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 01:05:20 2019

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import cv2

MAX_KERNEL_LENGTH = 31

a1 = np.zeros((100,100))
a2 = np.zeros((100,100)) + 100
a3 = np.zeros((100,100)) + 200

a4 = np.concatenate((a1,a2),axis=1)
myPlot1 = np.concatenate((a4,a3),axis=1)

myPlot2 = myPlot1 + 50 * (np.random.rand(100, 300) - 0.5)

plt.figure(1)
plt.imshow(myPlot2,cmap="gray")

for i in range(1, MAX_KERNEL_LENGTH, 2):
    dst = cv2.GaussianBlur(myPlot2, (i, i), 0)

plt.figure(2)
plt.imshow(dst,cmap="gray")

image = cv2.imread("clouds.jpg")

for i in range(1, MAX_KERNEL_LENGTH, 2):
    dst1 = cv2.GaussianBlur(image, (i, i), 0)

cv2.imshow("Over the Clouds - blur", dst1)
    
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Over the Clouds", image)
cv2.imshow("Over the Clouds - gray", gray_image)
cv2.waitKey(0)
cv2.destroyAllWindows()