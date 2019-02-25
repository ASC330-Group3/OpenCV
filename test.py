# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:42:14 2019

@author: Samuel Gibbs
"""

import cv2

video = cv2.VideoCapture(1)
#video.set(cv2.CAP_PROP_FRAME_WIDTH, 1920) # set the resolution - 640,480
#video.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
        # Read a new frame
        ok, frame = video.read()
        cv2.imshow("Test", frame)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27 : 
            cv2.destroyAllWindows()
            break