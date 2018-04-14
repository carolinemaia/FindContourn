# -*- coding: utf-8 -*-

import cv2

biggest_contour = None
biggest_area = 1
imageFile='fachada.jpg'
image = cv2.imread(imageFile)
height, width, channels = image.shape
imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
imgray = cv2.GaussianBlur(imgray, (5, 5), 0)

# O código abaixo define um retângulo na região central com 20% do comprimento e 20% de altura
MIN_THRESH = 0
CENTRAL_AREA = 0.25
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


while True:#Loop que processa a imagem aumentando o MIN_THRESH ou diminuindo CENTRAL_AREA 
	if(CENTRAL_AREA<0.01):
		break #interrompe a busca se a área central estiver tomando a imagem toda
	
	_, thresh = cv2.threshold(imgray, MIN_THRESH, 255, cv2.THRESH_BINARY)
	_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	MIN_THRESH += 10 #aumenta o threshold mínimo para incluir mais áreas

	if(contours!=[]): #Verifica se detectou algum contorno		
		cnt = max(contours, key=cv2.contourArea) #pega o maior contorno
		
		#Extrai os quatro pontos extremos de um contorno
		extL = tuple(cnt[cnt[:, :, 0].argmin()][0])#mais a esquerda
		extR = tuple(cnt[cnt[:, :, 0].argmax()][0])#mais a direita
		extT = tuple(cnt[cnt[:, :, 1].argmin()][0])#mais alto
		extB = tuple(cnt[cnt[:, :, 1].argmax()][0])#mais baixo
		
		if (isExtCentral(extL,extR,extT,extB)): #verifica se os pontos estão dentro da área central
			area = cv2.contourArea(cnt)
			if(area>biggest_area):#salva o contorno de maior área
				biggest_area=area
				biggest_contour=cnt
			else:
				break #interrompe a busca se os contornos ficarem menores
		
	#se não encontrou nenhum contorno reinicia as buscas aumentando a região central
	if(MIN_THRESH>255): 
		MIN_THRESH = 0
		CENTRAL_AREA -= CENTRAL_AREA*0.5
		MIN_X = int(width * CENTRAL_AREA)
		MAX_X = int(width * (1 - CENTRAL_AREA))
		MIN_Y = int(height * CENTRAL_AREA)
		MAX_Y = int(height * (1 - CENTRAL_AREA))


#desenha quadrado que corresponde a região central
cv2.rectangle(image, (MIN_X, MIN_Y), (MAX_X, MAX_Y), (0, 255, 0), 3)

moments = cv2.moments(biggest_contour)
cx = int(moments['m10'] / biggest_area)
cy = int(moments['m01'] / biggest_area)
cv2.circle(image, (cx, cy), 5, (255, 255, 0), -1)  # desenha um ponto azul no centro das figuras selecionadas

cv2.drawContours(image, [biggest_contour], -1, (0, 255, 255), 2)  # desenha borda amarela na figura de maior área
cv2.imshow('Threshold', thresh)
cv2.imshow('Maior contorno', image)
cv2.waitKey()
cv2.destroyAllWindows()
