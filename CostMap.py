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
    def __init__(self,camera_option):
        self.video = cv2.VideoCapture(camera_option)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1);
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        
        print(width,height)
        
        ret, self.aruco_frame = self.video.read()
#        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set the resolution - 640,480
#        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        
    def get_new_frame(self):
        self.flat_list = []
        #ok, frame = self.video.read()
        frame = self.aruco_frame
        
        frame = cv2.flip(frame, 1)
     
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        #cv2.imshow("CostMap",thresh)
      
        return ((thresh.flatten()/2.55).astype(int))
    
    def get_transform(self):
        
        #aruco width and height
        aruco_dimensions = 80
        
        ret, self.aruco_frame = self.video.read()
        #print(frame.shape) #480x640
        # Our operations on the frame come here
        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)
        retval, gray = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)
      
    
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250) 
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
                aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
         
                
                #convert arena coordinates to mm
                scaling_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/aruco_dimensions
                #pts = np.array([[corners[i][0][0][0],(corners[i][0][0][1]]), [corners[i][0][1][0],corners[i][0][1][1]] , [corners[i][0][2][0],corners[i][0][2][1]] , [corners[i][0][3][0],corners[i][0][3][1]]], np.int32)
                #pts = pts.reshape((-1,1,2))
                #cv2.polylines(self.aruco_frame,[pts],True,(0,255,255))
                #print(scaling_factor)
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
             
                z = (-z)
        
                distance_aruco_to_platform_centre = math.sqrt((((217/2)-50)*scaling_factor)**2 + (((407/2)-50)*scaling_factor)**2)
                angle_offset = math.atan(((407/2)*scaling_factor)/((217/2)*scaling_factor)) - (math.pi)/2
                
                platform_center_x = int(aruco_x_coor + distance_aruco_to_platform_centre*math.cos(z-angle_offset))
                platform_center_y = int(aruco_y_coor - distance_aruco_to_platform_centre*math.sin(z-angle_offset))
                
                
                #print(platform_center_x,platform_center_y)
                
                cv2.circle(self.aruco_frame,(platform_center_x,platform_center_y), 1, (0,0,255), -1)
                
                
                #Draw rotated rectangle
                angle = -z#angle_offset
                x0 = platform_center_x
                y0 = platform_center_y
                height = 370*scaling_factor
                width = 420*scaling_factor
                b = math.cos(angle) * 0.7
                a = math.sin(angle) * 0.7
                pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
                pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
                pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
                pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))
            
                cv2.line(self.aruco_frame, pt0, pt1, (255, 255, 255), 1)
                cv2.line(self.aruco_frame, pt1, pt2, (255, 255, 255), 1)
                cv2.line(self.aruco_frame, pt2, pt3, (255, 255, 255), 1)
                cv2.line(self.aruco_frame, pt3, pt0, (255, 255, 255), 1)
                
                rect_corners = np.array([[pt0],[pt1],[pt2],[pt3]])
                
                cv2.fillPoly(self.aruco_frame,[rect_corners],(0,0,0))
               
                ###### DRAW ID #####
                #cv2.putText(frame, str(x) + "," + str(y), (int(x)+20,int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                #cv2.putText(self.aruco_frame, str((z/math.pi)*180), (int(x_coor)+20,int(y_coor)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
                # Display the resulting frame
                #return angle, x , y
                found = 1
                
                transform_dict = {
                        "state" : found,
                        "x" : platform_center_x,
                        "y" : platform_center_y,
                        "angle" : z
                        }
                
                return (transform_dict)
        else:
            found = 0
            transform_dict = {
                        "state" : found,
                        "x" : 0,
                        "y" : 0,
                        "angle" : 0
                        }
            return (transform_dict)
            

                
        
    def show_frame(self):
        #cv2.imshow('costmap',self.thresh)
        cv2.imshow('Aruco',self.aruco_frame)
        
        
    def stop(self):
        cv2.destroyAllWindows()
        self.video.release()
        
if __name__ == '__main__':
    map = map_capture(0)
    
    while 1:
       
        map.get_transform()
        map.get_new_frame()
        map.show_frame()
        k = cv2.waitKey(1) & 0xff
        #Press escape to close program and take a picture
        if k == 27 :
            map.stop()
            break