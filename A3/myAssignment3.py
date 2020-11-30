# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 12:42:53 2019

@author: Guillaume

"""

import cv2
import numpy as np
#import scipy
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

############# Create set of faces from pictures of internet

for i in range(0,10):

    image = cv2.imread('pictures_set/barack/' + str(i + 1) +'.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
    ) 
    
    
    print("[INFO] Found {0} Faces!".format(len(faces)))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        print("[INFO] Object found. Saving locally.")
        #cv2.imwrite('t_n' + str(i+1) + '_' + str(w) + '_' + str(h) + '_faces.jpg', roi_color)
        cv2.imwrite('b_' + str(i+1) + '.jpg', roi_color)
    
    status = cv2.imwrite('t_faces_detected_' + str(i+1) + '.jpg', image)
    print("[INFO] Image faces_detected.jpg written to filesystem: ", status)


## Resampling part
    
for i in range(0,10):

    oriimg = cv2.imread('clean_set/t_' + str(i+1) + '.jpg',cv2.IMREAD_COLOR)
    newimg = cv2.resize(oriimg,(140,140))
    cv2.imwrite('clean_Set/t_' + str(i+1) + '_r.jpg',newimg)

j = 0
myMatrix1 = np.zeros((20,140,140))

for char in ['b','t']:
    
    for i in range(0,10):
        
        myImg = cv2.imread('clean_Set/' + char +'_' + str(i + 1) + '_r.jpg')
        myGray = cv2.cvtColor(myImg, cv2.COLOR_BGR2GRAY)
        myMatrix1[i + 10*j,:,:] = myGray
        
    j=+1
   
########### Creation of 2D matrix
    
# Create m x d data matrix
m = 20
d = 140 * 140
X = np.reshape(myMatrix1, (m, d))

Xmean = np.mean(X, axis=0)

Xreal = X - Xmean

U, Sigma, VT = np.linalg.svd(Xreal, full_matrices=False)

# Sanity check on dimensions
print("X:", X.shape)
print("U:", U.shape)
print("Sigma:", Sigma.shape)
print("V^T:", VT.shape)

plt.figure(1)
plt.plot(range(0,20),Sigma)
plt.ylabel('sigma_i')
plt.xlabel('i')
plt.show()

myEigenFace = np.reshape(VT[1,:], (140, 140))
myMeanFace = np.reshape(Xmean, (140, 140))

#normalize for plot
data = myEigenFace

data = (data - data.min()) / (data.max() - data.min()) #normalizes data in range 0 - 255
data = 255 * data
img = data.astype(np.uint8)
img1 = cv2.resize(img,(800,800))
cv2.imshow("Window", img1)
#cv2.imshow('Mean pic',myEigenFace1)


num_components = 10 # Number of principal components
Y = np.matmul(Xreal, VT[:num_components,:].T)

plt.figure(3)
plt.plot(Y[:,0],Y[:,1],linestyle="",marker="o")
plt.ylabel('eigen_1')
plt.xlabel('eigen_0')
plt.show()

######### Reconstruct faces

MyFaceApprox1 = np.zeros((1,19600))
n_eigen = 15
image_id = 3

for i in range(0,n_eigen):
    coeff = np.matmul(Xreal[image_id,:], VT[i,:].T)
    a = VT[i,:] * coeff
    MyFaceApprox1 = MyFaceApprox1 + a
    

MyFaceApprox1 = MyFaceApprox1 + Xmean 
MyFaceApprox = np.reshape(MyFaceApprox1, (140, 140))
img2 = MyFaceApprox.astype(np.uint8)
img3 = cv2.resize(img2,(800,800))
cv2.imshow("Face Approximation", img3)

##### CLUSTERING


kmeans = KMeans(n_clusters=2, init= 'k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto').fit(Y)

print(kmeans.labels_)