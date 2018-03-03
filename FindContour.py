# -*- coding: utf-8 -*-

import cv2
import numpy as np

maior_area = 0
maior_contorno = 0
img2 = cv2.imread('src.jpg')
height, width, channels = img2.shape
 
imgray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)  
imgray = cv2.GaussianBlur(imgray, (5, 5), 0)
_, thresh = cv2.threshold(imgray, 40, 255, cv2.THRESH_BINARY)

_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# O código abaixo define um retângulo na região central com 50% do comprimento e 50% de altura
CENTRAL_AREA=0.25
MIN_X=int(width*CENTRAL_AREA)
MAX_X=int(width*(1-CENTRAL_AREA))
MIN_Y=int(height*CENTRAL_AREA)
MAX_Y=int(height*(1-CENTRAL_AREA))

#desenha retângulo central em verde (devemos descartar o que está fora dele)
cv2.rectangle(img2, (MIN_X, MIN_Y),(MAX_X,MAX_Y),(0, 255, 0),3)

contour_sizes = [(cv2.contourArea(cnt), cnt) for cnt in contours]
biggest_contour=contours[0]
biggest_area=0
for cnt in contours:
    moments = cv2.moments(cnt)
    contour_area = moments['m00']
    if contour_area != 0:
        cx = int(moments['m10'] / contour_area)
        cy = int(moments['m01'] / contour_area)
        #print cx,cy #centro da forma geométrica detectada
        if(cx>MIN_X and cx<MAX_X and cy>MIN_Y and cy<MAX_Y): #só seleciona quem está dentro do retângulo
			contour_area=cv2.contourArea(cnt)
			if(contour_area>biggest_area):
				biggest_contour=cnt #armazena o que possui maior área
				biggest_area=contour_area
			print cx,cy,contour_area
			circulo = cv2.circle(img2, (cx, cy), 5, (255, 255, 0), -1) #desenha um ponto azul no centro das figuras selecionadas


cv2.drawContours(img2, [biggest_contour], -1, (0, 255, 255), 2) #desenha borda amarela na figura de maior área
cv2.imshow('Maior contorno', img2)
cv2.waitKey()
cv2.destroyAllWindows()
