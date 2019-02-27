# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:04:42 2019

@author: Samuel Gibbs
"""

import cv2

video = cv2.VideoCapture(1)
video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # set the resolution - 640,480
video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
        # Read a new frame
        ok, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        cv2.imshow('costmap',thresh)
        
        k = cv2.waitKey(30) & 0xff
        #Press escape to close program and take a picture
        if k == 27 : 
            #Saves the image to the directory of this python script
            cv2.imwrite( "CostMap2.jpg", thresh );
            cv2.destroyAllWindows()
            video.release()
            break