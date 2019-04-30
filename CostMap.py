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

        self.position_list = [ {"Clear_Commands":1,},
                            {
                                "pickup":"Red",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "pickup":"Green",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "pickup":"Blue",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 1",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 2",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 3",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "arm":"arm",
                                "coor_x":0,
                                "coor_y":0,
                            },
                ]
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
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
            parameters =  aruco.DetectorParameters_create()

            #lists of ids and the corners beloning to each ids
            corners, ids, rejectedImgPoints = aruco.detectMarkers(webcam_feed, aruco_dict, parameters=parameters)
            #print(corners)

            webcam_feed = aruco.drawDetectedMarkers(webcam_feed, corners,ids,(255,255,0))
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
    
    def get_position_list(self):
        return (self.position_list)

    def detect_triangle(self,c):
        # initialize the shape name and approximate the contour
         # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)
        conversion_factor =0.213
        act_hypot_length = 160
        act_tri_height = 80
        act_area = 430
        angle=0
        point=0
        # if the shape is a triangle, it will have 3 vertices
        
        if len(approx) == 3:          
            (x,y), (w, h), angle = cv2.minAreaRect(approx)
            #ar = w / float(h)
            
            area = cv2.contourArea(c)
         
            if (area >= act_area*0.8) and (area <= act_area*1.2):
                
                side_1 = math.sqrt((approx[0][0][0]-approx[1][0][0])**2+(approx[0][0][1]-approx[1][0][1])**2)/conversion_factor
                side_2 = math.sqrt((approx[0][0][0]-approx[2][0][0])**2+(approx[0][0][1]-approx[2][0][1])**2)/conversion_factor
                side_3 = math.sqrt((approx[1][0][0]-approx[2][0][0])**2+(approx[1][0][1]-approx[2][0][1])**2)/conversion_factor
                list_sides = [side_1,side_2,side_3]
                hypot = max(list_sides)
                idx = list_sides.index(hypot)
                if(idx == 0):
                    point = 2
                    mid_point_x = (approx[0][0][0]+approx[1][0][0])/2
                    mid_point_y = (approx[0][0][1]+approx[1][0][1])/2
                elif(idx== 1):
                    point = 1
                    mid_point_x = (approx[0][0][0]+approx[2][0][0])/2
                    mid_point_y = (approx[0][0][1]+approx[2][0][1])/2
                else:
                    point = 0
                    mid_point_x = (approx[2][0][0]+approx[1][0][0])/2
                    mid_point_y = (approx[2][0][1]+approx[1][0][1])/2
                    
                #cv2.circle(self.aruco_frame,(approx[point][0][0],approx[point][0][1]),5,(0,0,255),-1)
                #cv2.circle(self.aruco_frame,(int(mid_point_x),int(mid_point_y)),5,(0,0,255),-1)
                
                opp = (approx[point][0][1]-mid_point_y)
                adj = (approx[point][0][0]-mid_point_x)
                if (abs(adj) < 1e-6):
                        adj=0.0000001
                angle = math.atan2(opp,adj)
                print(angle*(180/math.pi))
#                    print(side_1,side_2,side_3)
                
                return (approx[point][0][0],approx[point][0][1],angle)
          
        return(-1,-1,-1)
        # return the name of the shape
           




    def get_transform(self):

        transform_dict = {
                            "state" : 0,
                            "x" : 0,
                            "y" : 0,
                            "angle" : 0,
                            }
        self.position_list = [
                            {"Clear_Commands":1,},
                            {
                                "pickup":"Red",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "pickup":"Green",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "pickup":"Blue",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 1",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 2",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 3",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "arm":"arm",
                                "coor_x":0,
                                "coor_y":0,
                            },
                ]

        ret, self.aruco_frame = self.video.read()
        if (ret == 0):
            return (transform_dict)

        cam_feed = self.aruco_frame.copy()
  
        aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_250)
        parameters =  aruco.DetectorParameters_create()

        #lists of ids and the corners beloning to each ids
        corners, ids, rejectedImgPoints = aruco.detectMarkers(self.aruco_frame, aruco_dict, parameters=parameters)
        #print(corners)

        self.aruco_frame = aruco.drawDetectedMarkers(self.aruco_frame, corners,ids,(255,255,0))
        cameraMatrix = np.array([[1.3953673275755928e+03, 0, 9.9285445205853750e+02], [0,1.3880458574466945e+03, 5.3905119245877574e+02],[ 0., 0., 1.]])
        distCoeffs = np.array([5.7392039180004371e-02, -3.4983260309560962e-02,-2.5933903577082485e-03, 3.4269688895033714e-03,-1.8891849772162170e-01 ])
        
        
        gray = cv2.cvtColor(cam_feed, cv2.COLOR_BGR2GRAY)
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY)[1]
        #cv2.imshow("thresh", thresh)
        # find contours in the thresholded image and initialize the
        # shape detector
        cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       
        triangle_found = 0
        # loop over the contours
        for c in cnts:
            point_x,point_y,angle = self.detect_triangle(c)
            #cv2.drawContours(self.aruco_frame, [c], -1, (0, 255, 0), 2)
            #cv2.circle(self.aruco_frame,(point_x,point_y),5,(0,0,255),-1)
            
            #self.detect_triangle(c)
            if (point_x != -1):
                
                cv2.drawContours(self.aruco_frame, [c], -1, (0, 255, 0), 2)
#                
                triangle_found = 1
                scaling_factor = 0.21242645786248002
                x0 = point_x
                y0 = point_y
                height = 500*scaling_factor
                width = 500*scaling_factor
                
                b = math.cos(angle) * 0.5
                a = math.sin(angle) * 0.5
                pt0 = (int(x0 - a * height - b * width), int(y0 + b * height - a * width))
                pt1 = (int(x0 + a * height - b * width), int(y0 - b * height - a * width))
                pt2 = (int(2 * x0 - pt0[0]), int(2 * y0 - pt0[1]))
                pt3 = (int(2 * x0 - pt1[0]), int(2 * y0 - pt1[1]))

                self.rect_corners = np.array([[pt0],[pt1],[pt2],[pt3]])

                cv2.fillPoly(self.aruco_frame,[self.rect_corners],(0,0,0))

                #The costmap image is flipped along the x -axis for screen coordinates:
                point_y = self.height - point_y
                angle = -angle
                update = {"state" : 1,
                            "x" : point_x,
                            "y" : point_y,
                            "angle" : angle}
                transform_dict.update(update)
                break
            
                
        if (triangle_found ==0):
            
            cv2.fillPoly(self.aruco_frame,[self.rect_corners],(0,0,0))
                
        
        if np.all(ids is not None):
            for i in range(0,int(ids.size)):
                
                
                
                if (ids[i][0]==20):
                    rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
                    (rvec-tvec).any() # get rid of that nasty numpy value array error
                    #aruco.drawAxis(self.aruco_frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
                    #aruco.drawDetectedMarkers(self.aruco_frame, corners) #Draw A square around the markers
                    aruco_x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
                    aruco_y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4

                    
                    x_ref = (corners[i][0][0][0] - corners[i][0][1][0])
                    y_ref = (corners[i][0][0][1] - corners[i][0][1][1])
                    if (abs(x_ref) < 1e-6):
                        x_ref=0.0000001

                    z = math.atan2(y_ref,x_ref)
                    z = (-z)
                    
                    #Arm Part-------------------------------------------------------------------------------------
                    marker_dimension = 180
                    conversion_factor = (math.sqrt((abs(corners[i][0][0][0] - corners[i][0][1][0]))**2+(abs(corners[i][0][0][1] - corners[i][0][1][1]))**2))/marker_dimension
                    distance_to_box = 205*conversion_factor
                    angle_offset = 0
                    arm_base_x,arm_base_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_box,z,angle_offset)

                    #cv2.circle(self.aruco_frame,(arm_base_x,arm_base_y),5,(255,255,255),-1)

                    arm_base_y = self.height - arm_base_y

                    update ={
                                "arm":"arm",
                                "coor_x":arm_base_x,
                                "coor_y":arm_base_y,
                            
                            }

                    self.position_list[7].update(update)
                    

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

                    distance_to_parking = 280*conversion_factor
                    parking_x,parking_y =  self.__transform_coordinates(red_x,red_y,distance_to_parking,z,angle_offset)

                    red_y = self.height - red_y
                    parking_y = self.height - parking_y
                    if (red_x != -1):

                        update =  {
                                "pickup":"Red",
                                "block_x":red_x,
                                "block_y":red_y,
                                "parking_x":parking_x,
                                "parking_y":parking_y,
                            }

                        self.position_list[1].update(update)

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
                    green_x,green_y=self.__get_obj_pos(box_centre_x,box_centre_y,box_area,z,cam_feed.copy())

                    distance_to_parking = 280*conversion_factor
                    parking_x,parking_y =  self.__transform_coordinates(green_x,green_y,distance_to_parking,z,angle_offset)

                    green_y = self.height - green_y
                    parking_y = self.height - parking_y
                    if (green_x != -1):

                        update =  {
                                "pickup":"Green",
                                "block_x":green_x,
                                "block_y":green_y,
                                "parking_x":parking_x,
                                "parking_y":parking_y,
                            }

                        self.position_list[2].update(update)

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
                    blue_x,blue_y=self.__get_obj_pos(box_centre_x,box_centre_y,box_area,z,cam_feed.copy())

                    distance_to_parking = 280*conversion_factor
                    parking_x,parking_y =  self.__transform_coordinates(blue_x,blue_y,distance_to_parking,z,angle_offset)

                    blue_y = self.height - blue_y
                    parking_y = self.height - parking_y
                    if (blue_x != -1):

                        update =  {
                                "pickup":"Blue",
                                "block_x":blue_x,
                                "block_y":blue_y,
                                "parking_x":parking_x,
                                "parking_y":parking_y,
                            }

                        self.position_list[3].update(update)


                if(ids[i][0]==14):
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

                    distance_to_parking = 280*conversion_factor
                    parking_x,parking_y =  self.__transform_coordinates(box_centre_x,box_centre_y,distance_to_parking,z,angle_offset)
                    parking_y = self.height - parking_y


                    update ={
                            "dropoff":"dropoff 1",
                            "dropoff_x":box_centre_x,
                            "dropoff_y":box_centre_y,
                            "parking_x":parking_x,
                            "parking_y":parking_y,
                            }

                    self.position_list[4].update(update)

                if(ids[i][0]==15):
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

                    distance_to_parking = 280*conversion_factor
                    parking_x,parking_y =  self.__transform_coordinates(box_centre_x,box_centre_y,distance_to_parking,z,angle_offset)

                    
                    parking_y = self.height - parking_y


                    update ={
                            "dropoff":"dropoff 2",
                            "dropoff_x":box_centre_x,
                            "dropoff_y":box_centre_y,
                            "parking_x":parking_x,
                            "parking_y":parking_y,
                            }

                    self.position_list[5].update(update)

                if(ids[i][0]==14):
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

                    distance_to_parking = 280*conversion_factor
                    parking_x,parking_y =  self.__transform_coordinates(box_centre_x,box_centre_y,distance_to_parking,z,angle_offset)

                    
                    parking_y = self.height - parking_y

                    update ={
                            "dropoff":"dropoff 3",
                            "dropoff_x":box_centre_x,
                            "dropoff_y":box_centre_y,
                            "parking_x":parking_x,
                            "parking_y":parking_y,
                            }

                    self.position_list[6].update(update)


                if(ids[i][0]==20):
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

                    distance_to_box = 205*conversion_factor
                    angle_offset = 0
                    arm_base_x,arm_base_y = self.__transform_coordinates(aruco_x_coor,aruco_y_coor,distance_to_box,z,angle_offset)

                    #cv2.circle(self.aruco_frame,(arm_base_x,arm_base_y),5,(255,255,255),-1)

                    arm_base_y = self.height - arm_base_y

                    update ={
                                "arm":"arm",
                                "coor_x":arm_base_x,
                                "coor_y":arm_base_y,
                            
                            }

                    self.position_list[7].update(update)


       
        


            update = {
                        "state" : 0,
                        "x" : 0,
                        "y" : 0,
                        "angle" : 0,
                    }
            transform_dict.update(update)

            self.position_list = [ {"Clear_Commands":1,},
                            {
                                "pickup":"Red",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "pickup":"Green",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "pickup":"Blue",
                                "block_x":0,
                                "block_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 1",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 2",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "dropoff":"dropoff 3",
                                "dropoff_x":0,
                                "dropoff_y":0,
                                "parking_x":0,
                                "parking_y":0,
                            },
                            {
                                "arm":"arm",
                                "coor_x":0,
                                "coor_y":0,
                            },
                ]


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
            #cv2.circle(output,(cX,cY),5,(0,255,0),-1)
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
            #cv2.circle(img,(cX,cY),5,(0,255,0),-1)

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
    map = map_capture(1)

    while 1:


        #map.get_transform()

        ret, frame = map.get_webcam_feed()
        if (ret==1):
            x = map.get_position_list()
            
            print(map.get_transform())
            print(x[6])
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