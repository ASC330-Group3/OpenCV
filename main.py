# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:31:27 2019

@author: Samuel Gibbs
"""

import cv2
import sys
import numpy as np




if __name__ == '__main__' :
    boolObjectDetected = False
    boolDebugMode = False
    boolPause = False
    
    video = cv2.VideoCapture(0)
    video.set(3, 640) # set the resolution
    video.set(4, 480)
    
    img = cv2.imread('TestImage.jpg',0)
    
    # Set up the detector with default parameters.
    detector = cv2.SimpleBlobDetector()    
    
    
#    
#    kernelErode = np.ones((3,3),np.uint8)
#    kernelDilate = np.ones((5,5),np.uint8)
#    
    imgThreshold = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,115,20)
#    #ret,imgThreshold = cv2.threshold(img,100,255,cv2.THRESH_TOZERO)
#    
#    erotion = cv2.erode(imgThreshold,kernelErode,iterations = 1)
#    dilation = cv2.dilate(erotion,kernelDilate,iterations = 2)
#
    # Detect blobs.
    keypoints = detector.detect(img)
     
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
    cv2.imshow("Blob Detection",im_with_keypoints)
    

#    cv2.imshow("Grayscale", img)
#    cv2.imshow("imgThreshold", imgThreshold)
#    cv2.imshow("erotion3", erotion)
#    cv2.imshow("dilation", dilation)

    
    cv2.waitKey(0)
    k = cv2.waitKey(100) & 0xff
    if k == 27 : 
        cv2.destroyAllWindows()
#    if not video.isOpened():
#        print ("Could not open video")
#        sys.exit()
# 
#    # Read first frame.
#    ok, frame = video.read()
#    if not ok:
#        print ('Cannot read video file')
#        sys.exit()
#        
#    while True:
#        ok, frame1 = video.read()
#        frameGray = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
#        
#        ret,thresh1 = cv2.threshold(frameGray,127,255,cv2.THRESH_TRUNC )
#        
#        timer = cv2.getTickCount()
# 
# 
#        # Calculate Frames per second (FPS)
#        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
# 
#        # Display FPS on frame
#        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
#
#        cv2.imshow("Test", thresh1)
#        k = cv2.waitKey(100) & 0xff
#        if k == 27 : break
#        
#    cv2.destroyWindow('Test')