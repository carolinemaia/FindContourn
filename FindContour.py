#-*- coding: utf-8 -*-
import cv2
import numpy as np

##maior_area = 0
##maior_contorno = 0
img2 = cv2.imread('fachada.jpg')  
height, width, channels = img2.shape 

imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) 
imgray = cv2.GaussianBlur(imgray, (5, 5), 0)

#_, thresh = cv2.threshold(imgray, 40, 25, cv2.THRESH_BINARY)
thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, \
                                  cv2.THRESH_BINARY, 11,2)
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


CENTRAL_AREA = 0.25
MIN_X=int(width*CENTRAL_AREA)
MAX_X=int(width*(1-CENTRAL_AREA))
MIN_Y=int(height*CENTRAL_AREA)
MAX_Y=int(height*(1-CENTRAL_AREA))


#cv2.rectangle(img2, (MIN_X, MIN_Y),(MAX_X,MAX_Y),(0,255,0),2) #VERDE

contour_sizes = [(cv2.contourArea(cnt), cnt) for cnt in contours]
biggest_contour=contours[0]
biggest_area=0
for cnt in contours: 
    moments = cv2.moments(cnt)  
    contour_area = moments['m00']
    if contour_area != 0:
        cx=int(moments['m10']/contour_area)
        cy=int(moments['m01']/contour_area)
        if(cx>MIN_X and cx<MAX_X and cy>MIN_Y and cy<MAX_Y):
            contour_area=cv2.contourArea(cnt)
            if(contour_area>biggest_area):
                biggest_contour=cnt
                biggest_area=contour_area
            print cx,cy,contour_area 
            #circulo = cv2.circle(img2, (cx,cy),5,(255,255,0), -1) 

cv2.drawContours(img2, [biggest_contour], -1, (0, 255, 255), 2) 
cv2.imshow('Maior contorno', img2)
cv2.waitKey()
cv2.destroyAllWindows()
