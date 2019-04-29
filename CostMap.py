# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:04:42 2019
@author: Samuel Gibbs
"""
#git test

import cv2
import cv2.aruco as aruco

import math
import numpy as np
import time

class map_capture():
    def __init__(self,camera_option):
        self.camera_option = camera_option
        self.video = cv2.VideoCapture(self.camera_option)        
        self.video.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.width = self.video.get(cv2.CAP_PROP_FRAME_WIDTH)   # float
        self.height = self.video.get(cv2.CAP_PROP_FRAME_HEIGHT) # float
        self.rect_corners = np.array([[0,0],[0,0],[0,0],[0,0]])

        
        ret, self.aruco_frame = self.video.read()
        #self.smooth_plat_coor_x = [0]
        #self.smooth_plat_coor_y = [0]
        #self.smooth_plat_angle = [0]
#        self.video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) # set the resolution - 640,480
#        self.video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    def get_webcam_feed(self):
        ret,webcam_feed = self.video.read()

        if (ret == 0):
            return(ret,0)

            #self.video.release()
        else:
            return (ret,webcam_feed)

    def get_new_frame(self):
        self.flat_list = []

        frame = self.aruco_frame.copy()

        frame = cv2.flip(frame, 0)

        #cv2.imshow('flip',frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
        #cv2.imshow("CostMap",thresh)

        return ((thresh.flatten()/2.55).astype(int))

    def get_transform(self):

        transform_dict = {
                            "state" : 0,
                            "x" : 0,
                            "y" : 0,
                            "angle" : 0,
                            "station_state":0,
                            "station_x":0,
                            "station_y":0,
                            "destination_state":0,
                            "destination_x":0,
                            "destination_y":0,
                            "red_state": 0,
                            "red_x":0,
                            "red_y":0,
                            "green_state":0,
                            "green_x":0,
                            "green_y":0,
                            "blue_state":0,
                            "blue_x":0,
                            "blue_y":0,
                            "arm_state":0,
                            "arm_base_x":0,
                            "arm_base_y":0,
                            
                            }
        
        ret, self.aruco_frame = self.video.read()
        if (ret == 0):
            return (transform_dict)
        
        cam_feed = self.aruco_frame.copy()

        gray = cv2.cvtColor(self.aruco_frame, cv2.COLOR_BGR2GRAY)
        retval, gray = cv2.threshold(gray,250,255,cv2.THRESH_BINARY)

        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        parameters =  aruco.DetectorParameters_create()

        #lists of ids and the corners beloning to each ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(self.aruco_frame, aruco_dict, parameters=parameters)
        #print(corners)

        self.aruco_frame = aruco.drawDetectedMarkers(self.aruco_frame, corners,ids,(255,255,0))
        cameraMatrix = np.array([[1.3953673275755928e+03, 0, 9.9285445205853750e+02], [0,1.3880458574466945e+03, 5.3905119245877574e+02],[ 0., 0., 1.]])
        distCoeffs = np.array([5.7392039180004371e-02, -3.4983260309560962e-02,-2.5933903577082485e-03, 3.4269688895033714e-03,-1.8891849772162170e-01 ])
        if np.all(ids is not None):
            for i in range(0,int(ids.size)):
                
                if (ids[i][0]==0):
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    #(rvec-tvec).any() # get rid of that nasty numpy value array error


                    #aruco.drawAxis(self.aruco_frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                    #aruco.drawDetectedMarkers(self.aruco_frame, corners) #Draw A square around the markers
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4


                    #convert arena coordinates to mm
                    #scaling_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/aruco_dimensions
                    scaling_factor = 0.21242645786248002           

                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001
                    
                    z = math.atan2(y_ref,x_ref)
                    z = (-z)
                    
                    distance_aruco_to_platform_centre = 120*scaling_factor#math.sqrt((((370/2)-distance_to_edge)*scaling_factor)**2 + (((420/2)-distance_to_edge)*scaling_factor)**2)
                    angle_offset = 0#math.atan(((420/2)*scaling_factor)/((370/2)*scaling_factor)) - (math.pi)/2

                    y_offset = 0;
                    platform_center_x = int(aruco_x_coor + distance_aruco_to_platform_centre*math.cos(z-angle_offset))
                    platform_center_y = int((aruco_y_coor + y_offset) - distance_aruco_to_platform_centre*math.sin(z-angle_offset))

                    cv2.circle(self.aruco_frame,(platform_center_x,platform_center_y), 1, (0,0,255), -1)


                    #Draw rotated rectangle
                    angle = -z#angle_offset
                    x0 = platform_center_x
                    y0 = platform_center_y
                    height = 410*scaling_factor
                    width = 420*scaling_factor
                    b = math.cos(angle) * 0.9
                    a = math.sin(angle) * 0.9
                    pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
                    pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
                    pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
                    pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

                    self.rect_corners = np.array([[pt0],[pt1],[pt2],[pt3]])

                    cv2.fillPoly(self.aruco_frame,[self.rect_corners],(0,0,0))

                    #The costmap image is flipped along the x -axis for screen coordinates:
                    platform_center_y = self.height - platform_center_y
                    
                    
                    update = {"state" : 1,
                            "x" : platform_center_x,
                            "y" : platform_center_y,
                            "angle" : z}

                    transform_dict.update(update)

                    
                else:
                    cv2.fillPoly(self.aruco_frame,[self.rect_corners],(0,0,0))

            
                if (ids[i][0]==4):

                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value arrayp error
                    
                    #aruco.drawAxis(self.aruco_frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                    #aruco.drawDetectedMarkers(cam_feed, corners) #Draw A square around the markers
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4


                    #convert arena coordinates to mm
                    #conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/aruco_dimensions
                    conversion_factor = 0.21242645786248002
                    
                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001
                    
                    z = math.atan2(y_ref,x_ref)
                    z = (-z)

                    distance_to_station = self.__get_distance_from_coor(500,0,0,conversion_factor)
                    angle_offset = 0
                    station_centre_x,station_centre_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_station,z,angle_offset)

                    station_centre_y = self.height - station_centre_y
                    
                    
                    update = {
                            "station_state" : 1,
                            "station_x" : station_centre_x,
                            "station_y" : station_centre_y,
                            }
                    
                    
                    transform_dict.update(update)
                
            
                if (ids[i][0]==6):

                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value array error
                    
                    #aruco.drawAxis(cam_feed, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                    #aruco.drawDetectedMarkers(cam_feed, corners) #Draw A square around the markers
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4

                    aruco_y_coor = self.height - aruco_y_coor
                    found = 1
                    update = {
                            "destination_state" : found,
                            "destination_x" : aruco_x_coor,
                            "destination_y" : aruco_y_coor,
                            }
                    transform_dict.update(update)
                    
                    
                    
                if(ids[i][0]==11):
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value arrayp error
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
                    #convert arena coordinates to mm
                    marker_dimension = 180
                    conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/marker_dimension
                    #conversion_factor =0.21242645786248002#coor per mm = 4.7mm to a coor
                
                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001
                    
                    z = math.atan2(y_ref,x_ref)
                    z = (-z)

                    distance_to_box = 140*conversion_factor
                    angle_offset = -math.pi/2
                    box_centre_x,box_centre_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_box,z,angle_offset)
                    box_area = 80*(conversion_factor)
                    red_x,red_y=self.__get_obj_pos(box_centre_x,box_centre_y,box_area,z,cam_feed.copy())

                    red_y = self.height - red_y
                    
                    if (red_x != -1):
                        update = {
                            "red_state" : 1,
                            "red_x" : red_x,
                            "red_y" : red_y,
                            }
                    
                        transform_dict.update(update)
                
                if(ids[i][0]==12):
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value arrayp error
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
                    #convert arena coordinates to mm
                    marker_dimension = 180
                    conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/marker_dimension
                    #conversion_factor =0.21242645786248002#coor per mm = 4.7mm to a coor
                
                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001
                    
                    z = math.atan2(y_ref,x_ref)
                    z = (-z)

                    distance_to_box = 140*conversion_factor
                    angle_offset = -math.pi/2
                    box_centre_x,box_centre_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_box,z,angle_offset)
                    box_area = 80*(conversion_factor)
                    red_x,red_y=self.__get_obj_pos(box_centre_x,box_centre_y,box_area,z,cam_feed.copy())

                    red_y = self.height - red_y
                    
                    if (red_x != -1):
                        update = {
                            "green_state" : 1,
                            "green_x" : red_x,
                            "green_y" : red_y,
                            }
                    
                        transform_dict.update(update)
                        
                if(ids[i][0]==13):
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value arrayp error
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
                    #convert arena coordinates to mm
                    marker_dimension = 180
                    conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/marker_dimension
                    #conversion_factor =0.21242645786248002#coor per mm = 4.7mm to a coor
                
                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001
                    
                    z = math.atan2(y_ref,x_ref)
                    z = (-z)

                    distance_to_box = 140*conversion_factor
                    angle_offset = -math.pi/2
                    box_centre_x,box_centre_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_box,z,angle_offset)
                    box_area = 80*(conversion_factor)
                    red_x,red_y=self.__get_obj_pos(box_centre_x,box_centre_y,box_area,z,cam_feed.copy())

                    red_y = self.height - red_y
                    
                    if (red_x != -1):
                        update = {
                            "blue_state" : 1,
                            "blue_x" : red_x,
                            "blue_y" : red_y,
                            }
                    
                        transform_dict.update(update)
                        
                if(ids[i][0]==20):
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value arrayp error
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
                    #convert arena coordinates to mm
                    marker_dimension = 100
                    conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/marker_dimension
                    #conversion_factor =0.21242645786248002#coor per mm = 4.7mm to a coor
                
                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001
                    
                    z = math.atan2(y_ref,x_ref)
                    z = (-z)

                    distance_to_box = 130.5*conversion_factor
                    angle_offset = -math.pi/2
                    arm_base_x,arm_base_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_box,z,angle_offset)
                 
                    cv2.circle(self.aruco_frame,(arm_base_x,arm_base_y),5,(255,255,255),-1)
                    
                    arm_base_y = self.height - arm_base_y
                    
                    update = {
                        "arm_state" : 1,
                        "arm_base_x" : arm_base_x,
                        "arm_base_y" : arm_base_y,
                        }
                    
                    transform_dict.update(update)
                    
                        
        else:
            cv2.fillPoly(self.aruco_frame,[self.rect_corners],(0,0,0))

          
            update = {
                        "state" : 0,
                        "x" : 0,
                        "y" : 0,
                        "angle" : 0,
                        "station_state":0,
                        "station_x":0,
                        "station_y":0,
                        "destination_state":0,
                        "destination_x":0,
                        "destination_y":0,
                        "red_state": 0,
                        "red_x":0,
                        "red_y":0,
                        "green_state":0,
                        "green_x":0,
                        "green_y":0,
                        "blue_state":0,
                        "blue_x":0,
                        "blue_y":0,
                        "arm_state":0,
                        "arm_base_x":0,
                        "arm_base_y":0,
                        
                    }
            transform_dict.update(update)
        
        #print(transform_dict)
        return (transform_dict)


##############################################################################################################

    def __get_obj_pos(self,box_centre_x,box_centre_y,box_area,angle,cam_feed):
        
        mask_ROI = np.zeros((int(self.height),int(self.width)), np.int8)


         #Draw rotated rectangle
        angle = -angle#angle_offset
        x0 = box_centre_x
        y0 = box_centre_y

        b = math.cos(angle) * 0.5
        a = math.sin(angle) * 0.5
        pt0 = (int(x0 - a * box_area - b * box_area), int(y0 + b * box_area - a * box_area))
        pt1 = (int(x0 + a * box_area - b * box_area), int(y0 - b * box_area - a * box_area))
        pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
        pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

        rect_corners = np.array([[pt0],[pt1],[pt2],[pt3]])

        cv2.fillPoly(mask_ROI,[rect_corners],(255,255,255))       

        mask_ROI = cv2.inRange(mask_ROI, 1, 255)

        output = cv2.bitwise_and(cam_feed,cam_feed, mask=mask_ROI)
        
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        retval, thresh = cv2.threshold(gray,50,255,cv2.THRESH_BINARY)
        
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(thresh,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)

        if cv2.countNonZero(dilation) == 0:
            return (-1,-1)
        
        else:
            M = cv2.moments(dilation)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output,(cX,cY),5,(0,255,0),-1)
        #cv2.imshow("output",img)
            return (cX,cY)

      

    def reconnect_camera(self):
        
        self.video.release()
        
        time.sleep(0.10)
        self.video = cv2.VideoCapture(self.camera_option)
                
    
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

    def __set_Arm_ROI(self,x,y,conversion_factor,arm_feed,radius):

        mask_ROI = np.zeros((int(self.height),int(self.width)), np.int8)
        arm_limits = int(conversion_factor*radius)
        mask_ROI = cv2.circle(mask_ROI,(int(x),int(y)),arm_limits,(255,255,255),-1)

        mask_ROI = cv2.inRange(mask_ROI, 1, 255)

        output = cv2.bitwise_and(arm_feed.copy(),arm_feed.copy(), mask=mask_ROI)
        return output

    def __detect_colour(self,arm_ROI,colour_block):

        img= arm_ROI.copy()
        hsv_frame = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

        #cv2.imshow("hsv",hsv_frame)

        cX = -1
        cY = -1
        if (colour_block ==1):

            #Red colours
            lower = np.array([0,62,190])
            upper = np.array([9,97,255])
        elif(colour_block ==2):
            #blue colours
            lower = np.array([88,70,186])
            upper = np.array([144,255,255])
        else:
            #green colours
            lower = np.array([0,0,234])
            upper = np.array([183,53,255])

        output = cv2.inRange(hsv_frame, lower, upper)
        kernel = np.ones((5,5),np.uint8)
        erosion = cv2.erode(output,kernel,iterations = 1)
        dilation = cv2.dilate(erosion,kernel,iterations = 1)


        if cv2.countNonZero(dilation) == 0:
            print("No objects detected")
        else:
            M = cv2.moments(dilation)

            # calculate x,y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(img,(cX,cY),5,(0,255,0),-1)

        #cv2.imshow("output",img)
        return (cX,cY)

    def arm_pickup_coor(self,colour_block):

        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        parameters =  aruco.DetectorParameters_create()
        ret, arm_feed = self.video.read()

        if (ret == True):

            #lists of ids and the corners beloning to each ids
            corners, ids, rejectedImgPoints = aruco.detectMarkers(arm_feed, aruco_dict, parameters=parameters)

            #arm_feed = aruco.drawDetectedMarkers(arm_feed, corners,ids,(255,255,0))
            cameraMatrix = np.array([[1.3953673275755928e+03, 0, 9.9285445205853750e+02], [0,1.3880458574466945e+03, 5.3905119245877574e+02],[ 0., 0., 1.]])
            distCoeffs = np.array([5.7392039180004371e-02, -3.4983260309560962e-02,-2.5933903577082485e-03, 3.4269688895033714e-03,-1.8891849772162170e-01 ])

            if np.all(ids != None):
                for i in range(0,int(ids.size)):
                    if (ids[i]==8):

                        rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                        (rvec-tvec).any() # get rid of that nasty numpy value array error

                        #aruco.drawAxis(arm_feed, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                        #aruco.drawDetectedMarkers(arm_feed, corners) #Draw A square around the markers
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


                        distance_to_arm = self.__get_distance_from_coor(500,0,0,conversion_factor)
                        angle_offset = 0
                        arm_centre_x,arm_centre_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_arm,z,angle_offset)



                        arm_ROI = self.__set_Arm_ROI(arm_centre_x,arm_centre_y,conversion_factor,arm_feed,350)



                        cX,cY = self.__detect_colour(arm_ROI,colour_block)
                        if (cX >-1):
                            found = True
                            distance_to_object = math.sqrt((arm_centre_y-cY)**2+(arm_centre_x-cX)**2)

                            angle = math.atan(abs(arm_centre_y-cY)/abs(arm_centre_x-cX))

                            x_arm = distance_to_object*math.cos(angle + z)
                            y_arm = distance_to_object*math.sin(angle + z)
                            x_arm_mm=(1/conversion_factor)*x_arm
                            y_arm_mm=(1/conversion_factor)*y_arm
                        else:
                            found = False
                            x_arm_mm,y_arm_mm= 0,0
                        
                        return(x_arm_mm,y_arm_mm,found)
            #cv2.circle(arm_feed,(arm_centre_x,arm_centre_y),1,(100,255,0),-1)
            #cv2.imshow("Arm feed",arm_feed)



#-------------------------------------------------------------------------------------------------------------------
    

    def show_frame(self):
        #cv2.imshow('costmap',self.thresh)
        cv2.imshow('Aruco',self.aruco_frame)


    def stop(self):
        cv2.destroyAllWindows()
        self.video.release()

if __name__ == '__main__':
    map = map_capture(0)

    while 1:
        
        
        #map.get_transform()
      
        ret, frame = map.get_webcam_feed()
        if (ret==1):
            
            print(map.get_transform())
            map.get_new_frame()
            map.show_frame()
            #cv2.imshow("webcam feed",frame)
            #print(map.arm_pickup_coor(2))
        else:
            map.reconnect_camera()
            print('Camera not connected')
        k = cv2.waitKey(1) & 0xff
        #Press escape to close program and take a picture
        if k == 27 :
            map.stop()
            break