# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:04:42 2019

@author: Samuel Gibbs
"""

import cv2  
import cv2.aruco as aruco

import math      
import numpy as np
import time

class map_capture():
    def __init__(self,camera_option):
        self.video = cv2.VideoCapture(camera_option)
        self.video.set(cv2.CAP_PROP_BUFFERSIZE, 1);
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.video.set(cv2.CAP_PROP_FOCUS, 0)
        #------Settings for big marker ID 0
        self.video.set(cv2.CAP_PROP_BRIGHTNESS,162)
        self.video.set(cv2.CAP_PROP_CONTRAST,255)
        self.video.set(cv2.CAP_PROP_SATURATION,255)
        self.video.set(cv2.CAP_PROP_SHARPNESS,255)
        #self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.75)

        #time.sleep(1)
        
        exposure = self.video.get(cv2.CAP_PROP_EXPOSURE)
        #self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
        
        self.video.set(cv2.CAP_PROP_EXPOSURE,exposure - 2)
        #------Settings for smaller ID 49 but requires smart exposure changes
#        self.video.set(cv2.CAP_PROP_BRIGHTNESS,90)
#        self.video.set(cv2.CAP_PROP_CONTRAST,0)
#        self.video.set(cv2.CAP_PROP_SHARPNESS,255)
#        
#        self.previous_exposure = 0;
        #self.set_camera_exposure()
        #---------------------------------
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
                       
        ret, self.aruco_frame = self.video.read()
#        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set the resolution - 640,480
#        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
    def set_camera_exposure(self):
        
        self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.75)
        time.sleep(1)
        exposure = self.video.get(cv2.CAP_PROP_EXPOSURE)
        self.video.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.25)
        
        print(exposure)
        if (self.previous_exposure != exposure):
        
            self.previous_exposure = exposure
            print('Camera Exposure = ' + str(exposure))
            if (exposure <= -5):
                print('Use Light Mode')
                self.video.set(cv2.CAP_PROP_EXPOSURE,-7)
            else:
                self.video.set(cv2.CAP_PROP_EXPOSURE,-3)
                print('Use Dark Mode')
        
    def get_webcam_feed(self):
        return self.webcam_feed
        
    def get_new_frame(self):
        self.flat_list = []
        
        frame = self.aruco_frame.copy()
        
        frame = cv2.flip(frame, 0)
     
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,85,255,cv2.THRESH_BINARY)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 1)
        dialation =cv2.dilate(erosion,kernel,iterations=1)
        #cv2.imshow("CostMap",dialation)
      
        return ((erosion.flatten()/2.55).astype(int))
    
    def __get_aruco_parameters(self):
        aruco_dimensions = 200
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250) 
        parameters =  aruco.DetectorParameters_create()
        return (aruco_dict,parameters,aruco_dimensions)
    
    def __set_Arm_ROI(self,x,y,conversion_factor,orignal_frame,radius):
        
        mask_ROI = np.zeros((int(self.height),int(self.width)), np.int8)
        arm_limits = int(conversion_factor*radius)
        mask_ROI = cv2.circle(mask_ROI,(int(x),int(y)),arm_limits,(255,255,255),-1)
        
        mask_ROI = cv2.inRange(mask_ROI, 1, 255)
   
        output = cv2.bitwise_and(orignal_frame,orignal_frame, mask=mask_ROI)
        return output
        
        
    def __get_distance_from_coor(self,x,y,offset_distance,conversion_factor):
        distance = math.sqrt((((y/2)-offset_distance)*conversion_factor)**2 + (((x/2)-offset_distance)*conversion_factor)**2)
        return distance        
    def __get_angle_offset(self,x,y,conversion_factor):
        offset = math.atan(((y/2)*conversion_factor)/((x/2)*conversion_factor)) - (math.pi)/2
        return offset
    
    def __transform_coordinates(self,x,y,distance,angle,angle_offset):
        new_x = int(x + distance*math.cos(angle-angle_offset))
        new_y = int(y - distance*math.sin(angle-angle_offset))
        
        return (new_x,new_y)
    
    def __detect_colour_in_arm_roi(self,arm_ROI):
        img= arm_ROI.copy()
        
        mask_ROI = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        retval, mask_thresh = cv2.threshold(mask_ROI,100,255,cv2.THRESH_BINARY)
        mask_thresh = cv2.inRange(mask_thresh, 1, 255)
        output = cv2.bitwise_and(img,img, mask=mask_thresh)
        hsv_frame = cv2.cvtColor(output,cv2.COLOR_BGR2HSV)
        lower_red = np.array([3,210,0])
        upper_red = np.array([7,255,255])
        output = cv2.inRange(hsv_frame, lower_red, upper_red)
        kernel = np.ones((3,3),np.uint8)
        erosion = cv2.erode(output,kernel,iterations = 1)
        
        im2, contours, hierarchy = cv2.findContours(erosion,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        cnt= contours[0][0]
        coor = (cnt[0][0],cnt[0][1])
        cv2.circle(img,coor,5,(0,255,0),-1)
        
        #cv2.imshow("output",img)
        #return output
    
    def get_transform(self):
       
        aruco_dict,parameters,aruco_dimensions = self.__get_aruco_parameters()
        
        ret, self.webcam_feed = self.video.read()
        self.aruco_frame = self.webcam_feed.copy()
        orignal_frame = self.webcam_feed.copy()

        #lists of ids and the corners beloning to each ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(self.aruco_frame, aruco_dict, parameters=parameters)
        
        #self.aruco_frame = aruco.drawDetectedMarkers(self.aruco_frame, corners,ids,(255,255,0))
        cameraMatrix = np.array([[1.3953673275755928e+03, 0, 9.9285445205853750e+02], [0,1.3880458574466945e+03, 5.3905119245877574e+02],[ 0., 0., 1.]])
        distCoeffs = np.array([5.7392039180004371e-02, -3.4983260309560962e-02,-2.5933903577082485e-03, 3.4269688895033714e-03,-1.8891849772162170e-01 ])
        if np.all(ids != None):
            for i in range(0,int(ids.size)):
                
                if ids[0][0] == 0: #49 is smaller id
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value array error
            
                    #aruco.drawAxis(self.aruco_frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                    #aruco.drawDetectedMarkers(self.aruco_frame, corners) #Draw A square around the markers
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
             
                    
                    #convert arena coordinates to mm
                    #conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/aruco_dimensions
                    conversion_factor = 0.21242645786248002
                    R = np.zeros(shape=(3,3))
                    cv2.Rodrigues(rvec[i-1  ], R, jacobian = 0)
                    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
             
                    singular = sy < 1e-6
                 
                    if  not singular :
                        z = math.atan2(R[1,0], R[0,0])
                    else :
                        z = 0
                 
                    z = (-z)
            
                    distance_marker_edge = 110
            
                    distance_aruco_to_platform_centre = self.__get_distance_from_coor(407,217,distance_marker_edge,conversion_factor)
                    angle_offset = self.__get_angle_offset(407,217,conversion_factor)
                    
                    platform_center_x,platform_center_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_aruco_to_platform_centre,z,angle_offset)
                    
                    #Draw rotated rectangle
                    angle = -z
                    x0 = platform_center_x
                    y0 = platform_center_y
                    height = 500*conversion_factor
                    width = 550*conversion_factor
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
                    
                    distance_to_arm = self.__get_distance_from_coor(400,0,0,conversion_factor)
                    angle_offset = 0
                    arm_centre_x,arm_centre_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_arm,z,angle_offset)
                    
                    #arm_ROI = self.__set_Arm_ROI(arm_centre_x,arm_centre_y,conversion_factor,orignal_frame,350)
                    #self.__detect_colour_in_arm_roi(arm_ROI)
                    
                    #The costmap image is flipped along the x -axis for screen coordinates:
                    platform_center_y = self.height - platform_center_y
                    
                    
                   
                    
                    
                    transform_dict = {
                            "state" : found,
                            "x" : platform_center_x,
                            "y" : platform_center_y,
                            "angle" : z
                            }
                
                    return (transform_dict)
        else:
            #\self.set_camera_exposure()
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
    map = map_capture(1)
    while 1:
        #map.set_camera_exposure()
        trans = map.get_transform()
        print(trans)
        map.get_new_frame()
        map.show_frame()
        k = cv2.waitKey(1) & 0xff
        #Press escape to close program and take a picture
        if k == 27 :
            map.stop()
            break