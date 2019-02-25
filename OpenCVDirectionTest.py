# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 14:08:45 2018

@author: Samuel Gibbs
"""

import cv2
import sys
import numpy as np
 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
if __name__ == '__main__' :
    

    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
        if tracker_type == 'MOSSE':
            tracker = cv2.TrackerMOSSE_create()
        if tracker_type == "CSRT":
            tracker = cv2.TrackerCSRT_create()
    
    # Read video
    
    video = cv2.VideoCapture(1)
    cv2.VideoCapture(cv2.CAP_DSHOW+1)
    #video.set(cv2.CAP_PROP_FOURCC, cv2.FOURCC('M', 'J', 'P', 'G'));
    video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # set the resolution - 640,480
    video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    
    # Exit if video not opened.
    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
 
    # Read first frame.
    ok, frame = video.read()
    if not ok:
        print ('Cannot read video file')
        sys.exit()
     
    kernelErode = np.ones((3,3),np.uint8)
    kernelDilate = np.ones((5,5),np.uint8)
        
    bbox = cv2.selectROI(frame, False)
    FPScounter = 0;
    
    ok = tracker.init(frame, bbox)
    objectTracker = 0;
 
    
    blobParams = cv2.SimpleBlobDetector_Params()    
    blobParams.minThreshold = 1;
    blobParams.maxThreshold = 10;
    
    # Filter by Area.
    blobParams.filterByArea = True
    blobParams.minArea = 1
      
    blobDetector = cv2.SimpleBlobDetector_create(blobParams)
    YellowPlatMin = np.array([10, 50, 50],np.uint8)
    YellowPlatMax = np.array([35, 255, 255],np.uint8)

    
    while True:
        # Read a new frame
        ok, frame = video.read()
        #frame = cv2.imread('TestImage.jpg')
        orignalFrame = frame
        #cv2.imshow("Livestream", frame)
        
        #Filter for yellow platform
        HSVFrame = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", HSVFrame)
        
        HSVThresholdFrame = cv2.inRange(HSVFrame, YellowPlatMin, YellowPlatMax)
        cv2.imshow("HSVThresholdFrame", HSVThresholdFrame)
        
        thresholdFrame = HSVThresholdFrame
        
        #grayFrame = cv2.cvtColor(HSVThresholdFrame, cv2.COLOR_BGR2GRAY)
        
        #retval, thresholdFrame = cv2.threshold(grayFrame,200,255,cv2.THRESH_BINARY)
        cv2.imshow("Threshold", thresholdFrame)
        
        erotion = cv2.erode(thresholdFrame,kernelErode,iterations = 1)
        dilation = cv2.dilate(erotion,kernelDilate,iterations = 3)
        
        keypoints = blobDetector.detect(dilation)
     
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(dilation, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        im2,contours,hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if len(contours) != 0:
            # draw in blue the contours that were founded
            #cv2.drawContours(orignalFrame, contours, -1, 255, 3)
        
            #find the biggest area
            cont = max(contours, key = cv2.contourArea)
            cv2.drawContours(orignalFrame, cont, -1, 255, 3)
            
            (x,y),(MA,ma),angle = cv2.fitEllipse(cont)
            x,y,w,h = cv2.boundingRect(cont)
            # draw the book contour (in green)
            rect = cv2.minAreaRect(cont)
            box = cv2.boxPoints(rect)
            
            box = np.int0(box)
            cv2.drawContours(orignalFrame,[box],0,(0,0,255),2)
            cv2.rectangle(orignalFrame,(x,y),(x+w,y+h),(0,255,0),2)
            
            Coor = ((x+w)/2,(y+h)/2)
            cv2.putText(frame, str(Coor),(100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            cv2.putText(frame, str(round(angle,2)),(100,100), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        #cv2.imshow("Result",dilation)

   
        #cv2.imshow("Image Manipulation", dilation)
        
        
        
        if not ok:
            break
         
        # Start timer
        timer = cv2.getTickCount()
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
 
        # Display FPS on frame
        
        if FPScounter == 1:
            cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);
 
    
        #ok, bbox = tracker.update(frame)
        
        if objectTracker == 1:
            if ok:
                # Tracking success
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                Coor = (int(bbox[0] + (bbox[2])/2), int(bbox[1] + (bbox[3])/2))
                
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
                cv2.putText(frame, str(Coor),(100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            else :
                # Tracking failure
                cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
     
            # Display tracker type on frame
            cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
         
        
        

        # Display result
        #cv2.imshow("Direction Test", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(30) & 0xff
        if k == 27 : 
            cv2.destroyAllWindows()
            break
        elif k == 102 :
            if FPScounter == 1:
                FPScounter = 0;
            else:
                FPScounter = 1;
        elif k == 116 :
            if objectTracker == 1:
                objectTracker = 0;
            else:
                objectTracker = 1;
        
    
            
            
            
            
            
            
            