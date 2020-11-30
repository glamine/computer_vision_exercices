# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:48:51 2019

@author: user
"""

import cv2
import numpy as np

print(cv2.__version__)

cap = cv2.VideoCapture('video.mpeg')

if cap.isOpened(): 
    ret, frame = cap.read()
else:
    ret = False
    print("Error opening video stream or file")    

fps = cap.get(cv2.CAP_PROP_FPS)
h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

# TO SAVE:
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('P','I','M', '1')
out = cv2.VideoWriter('output.mpeg',fourcc, fps, (int(w),int(h)))

frameNumber = 1


while cap.isOpened():
    ret1, frame = cap.read()
    
    if ret1:
        if cv2.waitKey(27) & 0xFF == ord('q'): 
            break
    else:
        break
    
    t = frameNumber/fps
    frame = cv2.GaussianBlur(frame, (3, 3), 0)
    src_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    
    if t < 5:#5: # original video
        cv2.imshow('frame',frame)
        out.write(frame)
    elif t < 10: # Sobel derivatives - edge detector
        
        scale = 1
        delta = 0
        ddepth = cv2.CV_16S
        
        # Gradient-X
        grad_x = cv2.Sobel(src_gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        grad_y = cv2.Sobel(src_gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    
    
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    
        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
    
        cv2.imshow('frame', grad)
        out.write(grad)
        
    elif t < 15: #Canny edges detector
        ratio = 3
        kernel_size = 3
        yf = 25
        m = yf/5
        p = -2*yf
        a = m*t + p
        low_threshold = 10 #a
        n = 5
        #print(a)
        
        bila = cv2.bilateralFilter(src_gray, n, 75, 75)
        bila = cv2.bilateralFilter(bila, n, 75, 75)
        
        detected_edges = cv2.Canny(bila, 0, low_threshold*ratio, kernel_size)
        mask = detected_edges != 0
        dst = frame * (mask[:,:,None].astype(frame.dtype))
        cv2.imshow('frame', dst)
        out.write(dst)
        
    elif t < 25: #25 #Hough Transform circles
        fac = 1/5
        circles = cv2.HoughCircles(src_gray,cv2.HOUGH_GRADIENT,1,w*fac,param1=100,param2=25,minRadius=int(w/10),maxRadius=int(w/3))
        
        if circles is None:
            cv2.imshow('frame', frame)
            out.write(frame)
            continue
        else:
            circles = np.uint16(np.around(circles))
            for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                # draw the center of the circle
                cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
                
            cv2.imshow('frame', frame)
            out.write(frame)
        
    else: #original
        cv2.imshow('frame', frame)
        out.write(frame)

    frameNumber += 1
    print(frameNumber)
    
cap.release()
out.release()
cv2.destroyAllWindows()
