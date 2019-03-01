# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:04:42 2019

@author: Samuel Gibbs
"""

import cv2        

class map_capture():
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 0);
#        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set the resolution - 640,480
#        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def get_new_frame(self):
        self.flat_list = []
        ok, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, self.thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
       
        return ((self.thresh.flatten()/2.55).astype(int))
        
        
    def show_frame(self):
        cv2.imshow('costmap',self.thresh)
        
    def stop(self):
        cv2.destroyAllWindows()
        self.video.release()
        
if __name__ == '__main__':
    map = map_capture()
    
    while 1:
        map.get_new_frame()
        map.show_frame()
        k = cv2.waitKey(30) & 0xff
        #Press escape to close program and take a picture
        if k == 27 :
            map.stop()
            break