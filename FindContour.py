# -*- coding: utf-8 -*-

import cv2
import numpy as np
import copy

biggest_contour = None
biggest_area = 1
#video
# cap = cv2.VideoCapture('video7.mp4')
cap = cv2.VideoCapture('video4.mp4')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# occluded = cv2.imread('interior2.jpg')
occluded = cv2.imread('interior.jpg')
occluded = cv2.resize(occluded,(width, height), interpolation = cv2.INTER_CUBIC)

debugContour=False

##imageFile='teste5.jpg'
##image = cv2.imread('test.jpg')
##height, width, channels = image.shape
##imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
##imgray = cv2.GaussianBlur(imgray, (5, 5), 0)

cannyFactor=4.5
CANNY_MIN_LIMIT = 25
CANNY_MAX_LIMIT = 75
silhouetteBlend = 3.0

# O código abaixo define um retângulo na região central com 20% do comprimento e 20% de altura
MIN_THRESH = 0
CENTRAL_AREA = 0.05
MIN_X = int(width * CENTRAL_AREA)
MAX_X = int(width * (1 - CENTRAL_AREA))
MIN_Y = int(height * CENTRAL_AREA)
MAX_Y = int(height * (1 - CENTRAL_AREA))


#verifica se os pontos extremos do contorno estão na região central
def isExtCentral(extL,extR,extT,extB):
	if(extL[0]<MIN_X or extR[0]>MAX_X or extT[1]<MIN_Y or extB[1]>MAX_Y):
		return False
	else:
		return True

def setCentralArea(area):
	global CENTRAL_AREA,MIN_X,MIN_Y,MAX_X,MAX_Y
	CENTRAL_AREA=area
	MIN_X = int(width * CENTRAL_AREA)
	MAX_X = int(width * (1 - CENTRAL_AREA))
	MIN_Y = int(height * CENTRAL_AREA)
	MAX_Y = int(height * (1 - CENTRAL_AREA))

setCentralArea(0.05)
while(cap.isOpened()):

	ret, frame = cap.read()
	if(ret==False):
		break

	MIN_THRESH = 0
	biggest_area=1

	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	while(MIN_THRESH<255):#Loop que processa a imagem aumentando o MIN_THRESH ou diminuindo CENTRAL_AREA

		blur = cv2.GaussianBlur(gray,(5,5),1)

		_, thresh = cv2.threshold(blur, MIN_THRESH, 255, cv2.THRESH_BINARY)
		MIN_THRESH += 10 #aumenta o threshold mínimo para incluir mais áreas
		_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

		selectedContour=[] #armazena apenas os contornos centrais
		if(contours!=[]): #Verifica se detectou algum contorno
			for cnt in contours:
				extL = tuple(cnt[cnt[:, :, 0].argmin()][0])#mais a esquerda
				extR = tuple(cnt[cnt[:, :, 0].argmax()][0])#mais a direita
				extT = tuple(cnt[cnt[:, :, 1].argmin()][0])#mais alto
				extB = tuple(cnt[cnt[:, :, 1].argmax()][0])#mais baixo

				if (isExtCentral(extL,extR,extT,extB)):
					selectedContour.append(cnt)

		if(selectedContour!=[]):
			cnt = max(selectedContour, key=cv2.contourArea) #seleciona o maior contorno

			area = cv2.contourArea(cnt)
			if(area>biggest_area):#salva o contorno de maior área
				biggest_area=area
				biggest_contour=cnt


	kernel = np.ones((5,5),np.uint8)
	cannyEdges = cv2.Canny(gray, CANNY_MIN_LIMIT * cannyFactor, CANNY_MAX_LIMIT * cannyFactor, apertureSize = 3)
	edges = cv2.dilate(cannyEdges,kernel,iterations = 2)
	edges = cv2.GaussianBlur(edges,(7,7),5)
	_,edges = cv2.threshold(edges,silhouetteBlend*255,0, cv2.THRESH_TRUNC)
	edges = cv2.bitwise_not(edges)
	edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)

	#Cria máscara para filtrar região do contorno
	mask = np.ones(frame.shape[:2], dtype="uint8") * 255
	cv2.drawContours(mask, [biggest_contour], -1, 0, -1)
	mask = cv2.bitwise_not(mask)

	#Seleciona silhuetas apenas na região do contorno
	silhouette=cv2.subtract(frame,edges)
	silhouette=cv2.bitwise_and(silhouette,silhouette,mask=mask)

	#Suaviza as bordas da região central
	maskSmooth = cv2.erode(mask,kernel,iterations = 2)
	maskSmooth = cv2.cvtColor(maskSmooth,cv2.COLOR_GRAY2RGB)
	maskSmooth = cv2.GaussianBlur(maskSmooth,(5,5),3)


	occludedDisplay = cv2.bitwise_and(occluded, occluded, mask=mask)
	occludedDisplay = cv2.subtract(occludedDisplay,silhouette)

	xRayView = cv2.subtract(frame,maskSmooth)
	xRayView = cv2.add(xRayView,occludedDisplay)
	xRayView = cv2.add(xRayView,silhouette)

	if(debugContour==True):
		#Desenha quadrado que corresponde a região central
		cv2.rectangle(frame, (MIN_X, MIN_Y), (MAX_X, MAX_Y), (0, 255, 0), 3)

		#Desenha um ponto azul no centro da figura selecionada
		moments = cv2.moments(biggest_contour)
		cx = int(moments['m10'] / biggest_area)
		cy = int(moments['m01'] / biggest_area)
		cv2.circle(frame, (cx, cy), 5, (255, 255, 0), -1)

		#Desenha borda amarela na figura de maior área
		cv2.drawContours(xRayView, [biggest_contour], -1, (0, 255, 255), 2)

	cv2.imshow('Frame',xRayView)
	if cv2.waitKey(50) & 0xFF == ord('q'): #Regula a velocidade de exibição do vídeo
		break


cap.release()
cv2.waitKey()
cv2.destroyAllWindows()
