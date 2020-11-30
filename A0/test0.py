# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 19:59:58 2019

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
import opencv as cv2

MAX_KERNEL_LENGTH = 31

myArray1 = np.zeros((100,100)) + 100
myArray2 = np.zeros((100,100)) + 150
myArray3 = np.zeros((100,100)) + 200

myPlot0 = np.concatenate((myArray1,myArray2),axis=1)
myPlot1 = np.concatenate((myPlot0,myArray3),axis=1)

myPlot2 = myPlot1 + 50 * (np.random.rand(100, 300) - 0.5)

plt.figure()
#plt.plot([1 2 3 4 5 6 7 8 9])
#plt.ion()
#plt.show()

plt.imshow(myPlot2,cmap="gray")

for i in range(1, MAX_KERNEL_LENGTH, 2):
    dst = cv2.GaussianBlur(myPlot2, (i, i), 0)

plt.imshow(dst,cmap="gray")