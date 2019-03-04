# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:04:42 2019

@author: Samuel Gibbs
"""

import cv2  
import cv2.aruco as aruco

import math      
import numpy as np

class map_capture():
    def __init__(self):
        self.video = cv2.VideoCapture(1)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1);
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set the resolution - 640,480
#        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def get_new_frame(self):
        self.flat_list = []
        ok, frame = self.video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
       
        return ((thresh.flatten()/2.55).astype(int))
        
    def get_transform(self):
        
        #Width and height of platform in mm
        object_width = 360
        object_height = 400
        
        #aruco width and height
        aruco_dimensions = 100
        
        ret, self.aruco_frame = self.video.read()
        #print(frame.shape) #480x640
        # Our operations on the frame come here
        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)
        #retval, gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    
        aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250) 
        parameters =  aruco.DetectorParameters_create()
     
        
        #lists of ids and the corners beloning to each ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
        #print(corners)
        
        self.aruco_frame = aruco.drawDetectedMarkers(self.aruco_frame, corners,ids,(255,255,0))
        cameraMatrix = np.array([[1.3953673275755928e+03, 0, 9.9285445205853750e+02], [0,1.3880458574466945e+03, 5.3905119245877574e+02],[ 0., 0., 1.]])
        distCoeffs = np.array([5.7392039180004371e-02, -3.4983260309560962e-02,-2.5933903577082485e-03, 3.4269688895033714e-03,-1.8891849772162170e-01 ])
        if np.all(ids != None):
            for i in range(0,int(ids.size)):
                
                rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                #(rvec-tvec).any() # get rid of that nasty numpy value array error
        
        
                aruco.drawAxis(self.aruco_frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                aruco.drawDetectedMarkers(self.aruco_frame, corners) #Draw A square around the markers
                x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
                
                
                
                #pts = np.array([[corners[i][0][0][0],(corners[i][0][0][1]]), [corners[i][0][1][0],corners[i][0][1][1]] , [corners[i][0][2][0],corners[i][0][2][1]] , [corners[i][0][3][0],corners[i][0][3][1]]], np.int32)
                #pts = pts.reshape((-1,1,2))
                #cv2.polylines(self.aruco_frame,[pts],True,(0,255,255))
                
                rotM = np.zeros(shape=(3,3))
                cv2.Rodrigues(rvec[i-1  ], rotM, jacobian = 0)
                R = rotM
                sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
         
                singular = sy < 1e-6
             
                if  not singular :
                    x = math.atan2(R[2,1] , R[2,2])
                    y = math.atan2(-R[2,0], sy)
                    z = math.atan2(R[1,0], R[0,0])
                else :
                    x = math.atan2(-R[1,2], R[1,1])
                    y = math.atan2(-R[2,0], sy)
                    z = 0
             
                
                #print(rotM)
        
                ###### DRAW ID #####
                #cv2.putText(frame, str(x) + "," + str(y), (int(x)+20,int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                #cv2.putText(self.aruco_frame, str((z/math.pi)*180), (int(x_coor)+20,int(y_coor)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                # Display the resulting frame
                #return angle, x , y
                found = 1
                return (z,x_coor,y_coor,found)
        else:
            found = 0
            return (0,0,0,found)
            

                
        
    def show_frame(self):
        #cv2.imshow('costmap',self.thresh)
        cv2.imshow('Aruco',self.aruco_frame)
        
        
    def stop(self):
        cv2.destroyAllWindows()
        self.video.release()
        
if __name__ == '__main__':
    map = map_capture()
    
    while 1:
        #map.get_new_frame()
        map.get_transform()
        map.show_frame()
        k = cv2.waitKey(1) & 0xff
        #Press escape to close program and take a picture
        if k == 27 :
            map.stop()
            break