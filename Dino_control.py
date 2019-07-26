#Imports
import cv2                                                                                                                                               
import numpy as np
import math
import keyboard

#Declaring varibles
background = None
accum_wt = 0.5
roi_top = 20
roi_bottom = 300
roi_right = 300
roi_left = 600

def calculateAngle(far, start, end):
    """Cosine rule"""
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2) / (2*b*c))
    return angle

def countFingers(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) > 3:
        defects = cv2.convexityDefects(contour, hull)
        cnt = 0
        if type(defects) != type(None):
            for i in range(defects.shape[0]):
                #Calculating the angle form the defects
                s, e, f, d = defects[i, 0]
                start = tuple(contour[s, 0])
                end = tuple(contour[e, 0])
                far = tuple(contour[f, 0])
                angle = calculateAngle(far, start, end)
                # Ignore the defects which are small and wide
                # Probably not fingers
                if d > 10000 and angle <= math.pi/2:
                    cnt += 1
        return (True, cnt)
    return (False, 0)
cam = cv2.VideoCapture(0)

while True:
    
    #Fetching frame form the camera
    _, frame = cam.read()
    
    #Extracting and marking the region of interest
    
    roi = frame[roi_top:roi_bottom, roi_right:roi_left]    
    cv2.rectangle(frame,(roi_left,roi_top),(roi_right,roi_bottom),(0,255,0),2)
    #Converting the color format
    hsv = cv2.cvtColor(cv2.medianBlur(roi,15),cv2.COLOR_BGR2HSV)    
    
    lower = np.array([0, 10, 20])     #Lower range of HSV color
    upper = np.array([30, 255, 255])  #Upper range of HSV color
    
    #Creating the mask
    mask = cv2.inRange(hsv,lower,upper)
    #Removing noise form the mask
    mask = cv2.dilate(mask,None,iterations=2)
    
    #Extracting contours form the mask
    cnts,_ = cv2.findContours(mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) > 0:
        for c in cnts:
            #To prevent the detection of the noise
            if cv2.contourArea(c) > 4000:
                #creating the convex hull around hand 
                hull = cv2.convexHull(c)
                cv2.drawContours(roi,[hull],0,(0,0,255),2)
                ret,cn= countFingers(c)
                if ret == True and cn<=2:
                    if cn == 0:
                        keyboard.press(" ")
                #Creating the bounding rectangel
    
    cv2.imshow("Live", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cam.release()
cv2.destroyAllWindows()
