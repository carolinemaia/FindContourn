import cv2
import numpy as np

maior_area = 0
maior_contorno = 0
img2 = cv2.imread('src.jpg')
height, width, channels = img2.shape
x1= int (width/4)
x2 = width-x1


img = img2[ 0:height, x1:x2]

 
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
_, thresh = cv2.threshold(imgray, 40, 255, cv2.THRESH_BINARY)


_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)



contour_sizes = [(cv2.contourArea(cnt), cnt) for cnt in contours]
biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]
for cnt in contours:
    moments = cv2.moments(cnt) 
    if moments['m00'] != 0:
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00'])
        moment_area = moments['m00']
        contour_area = cv2.contourArea(cnt)
        circulo = cv2.circle(img2, (0, 0), 5, (0, 255, 255), -1)



cv2.drawContours(img2, [biggest_contour], -1, (0, 255, 255), 2)
cv2.imshow('Maior contorno', img2)
cv2.waitKey()
cv2.destroyAllWindows()
