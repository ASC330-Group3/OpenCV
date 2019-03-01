# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 13:27:24 2019

@author: Samuel Gibbs
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 14:46:32 2019

@author: Samuel Gibbs
"""

import numpy as np
import cv2
import cv2.aruco as aruco

import math
#help(cv2.aruco)
 


cv_file = cv2.FileStorage(r"C:\Users/Samuel Gibbs/Documents/.Uni Year3/ACS330 - Group Project/OpenCV/Python/Camera Calibration/parameters.yaml", cv2.FILE_STORAGE_READ)

# Note : we also have to specify the type to retrieve otherwise we only get a
# FileNode object back instead of a matrix
cameraMatrix = cv_file.getNode("camera_matrix").mat()
distCoeffs = cv_file.getNode("dist_coeff").mat()


cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
#cap.set(15, -4)
#focus = 0  # min: 0, max: 255, increment:5
#cap.set(28, focus) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) # set the resolution - 640,480
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    #print(frame.shape) #480x640
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #retval, gray = cv2.threshold(gray,200,255,cv2.THRESH_BINARY)
    gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    cv2.imshow('Threshold',gray)
    cv2.imshow('Stream',frame)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters =  aruco.DetectorParameters_create()
 
    #print(parameters)
 
    '''    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
        '''
        #lists of ids and the corners beloning to each ids
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
    #print(corners)
 
    #It's working.
    # my problem was that the cellphone put black all around it. The alrogithm
    # depends very much upon finding rectangular black blobs
 
    
    frame = aruco.drawDetectedMarkers(frame, corners,ids,(255,255,0))
    
    if np.all(ids != None):
        for i in range(0,int(ids.size)):
            
            rvec, tvec,_ = aruco.estimatePoseSingleMarkers(corners[i], 0.05, cameraMatrix, distCoeffs) #Estimate pose of each marker and return the values rvet and tvec---different from camera coefficients
            #(rvec-tvec).any() # get rid of that nasty numpy value array error
    
    
            aruco.drawAxis(frame, cameraMatrix, distCoeffs, rvec[0], tvec[0], 0.1) #Draw Axis
            aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers
            x_coor = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2][0] + corners[i][0][3][0]) / 4
            y_coor = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2][1] + corners[i][0][3][1]) / 4
            
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
         
            print(np.array([x, y, z]))
            
            angle = (rvec[0][0]/math.pi)*180
            #print(rotM)
    
            ###### DRAW ID #####
            #cv2.putText(frame, str(x) + "," + str(y), (int(x)+20,int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
            cv2.putText(frame, str((z/math.pi)*180), (int(x_coor)+20,int(y_coor)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
            # Display the resulting frame
    cv2.putText(frame, "Id: " + str(ids), (0,64), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2,cv2.LINE_AA)
    #print(rejectedImgPoints)
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break
 
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()