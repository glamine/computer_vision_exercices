# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 12:45:29 2019

@author: guillaume
"""

import cv2
import numpy as np

cap = cv2.VideoCapture('video2.mpeg')

if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = cap.get(cv2.CAP_PROP_FPS)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print( length )

fourcc = cv2.VideoWriter_fourcc('P','I','M', '1')
out = cv2.VideoWriter('output2.mpeg',fourcc, fps, (int(w),int(h)))

for i in range(60):
    ret,background = cap.read()

frameNumber = 1
#lastFrameNumber = frameNumber

while(cap.isOpened()):
    _, frame = cap.read()
    
    t = frameNumber/30
    
    if t < 15 : 
        myGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',myGray)
        out.write(myGray)
        if 10 >= t > 5 :
            n = 2*(int(t) - 4)+1 
            myBlur = cv2.GaussianBlur(myGray, (n,n), 0)
            cv2.imshow('frame',myBlur) # blurred
            out.write(myBlur)
        elif t > 10:
            n=5
            if t < 12: # bilateral filter keeps edges 
                myBila = cv2.bilateralFilter(myGray, n, 75, 75) 
                cv2.imshow('frame',myBila) # blurred
                out.write(myBila)
            else :
               myBila = cv2.bilateralFilter(myGray, n, 75, 75)
               myBila = cv2.bilateralFilter(myBila, n, 75, 75) 
               myBila = cv2.bilateralFilter(myBila, n, 75, 75)
        
               cv2.imshow('frame',myBila) # blurred
               out.write(myBila)
            
    elif t < 20: 
        cv2.imshow('frame',frame) # normal
        out.write(frame)
        
    elif t < 50: 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) 
                
        lower_red = np.array([0,130,130])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv,lower_red,upper_red)
        lower_red = np.array([170,130,130])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv,lower_red,upper_red)
        mask1 = mask1+mask2
        
        red = cv2.bitwise_and(frame,frame, mask= mask1)  
        if t < 30:
            cv2.imshow('frame', red)
            out.write(red)

        elif t < 35: 

            kernel = np.ones((5,5),np.uint8)
            eros = cv2.erode(red,kernel,iterations = 1) # erosion
            cv2.imshow('frame', eros)
            out.write(eros)
            
        else: 
            mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel) # opening
            mask1 = cv2.dilate(mask1,kernel,iterations = 1) # dilatation
            mask2 = cv2.bitwise_not(mask1)
            
            res1 = cv2.bitwise_and(background,background,mask=mask1)
            res2 = cv2.bitwise_and(frame,frame, mask=mask2)
            magic = cv2.addWeighted(res1,1,res2,1,0)
            
            cv2.imshow('frame', magic)
            out.write(magic)
        
    else:
        if t < 55:
            # Canny edge detector
            if t < 53: # canny does not exist yet 
                canny = cv2.Canny(frame, 10, 30)
            else :
                canny = cv2.Canny(canny, 10, 30) # apply filter several times  --> NOT A LOT OF EDGES
            
            cv2.imshow('frame',canny) # edges
            out.write(canny)
    
    if cv2.waitKey(27) & 0xFF == ord('q'): 
        break

    frameNumber += 1
    print(frameNumber)
    
cap.release()
out.release()
cv2.destroyAllWindows()