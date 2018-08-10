# -*- coding: utf-8 -*-

import cv2
import numpy as np

cap = cv2.VideoCapture('video4.mp4')
WIDTH = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
HEIGHT = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

if (WIDTH*HEIGHT>768*432):
	HEIGHT=int((float(HEIGHT)/WIDTH)*768)
	WIDTH=768


print WIDTH,HEIGHT
HOUGH_WIDTH = 512#384#768#480
HOUGH_HEIGHT = 288#216#432#270

PERSPECTIVE_SMOOTH = 0.9#0.85
newPerspPts=[0,0,0,0]
perspLines=[0,np.pi/2,HOUGH_HEIGHT,np.pi/2]

occluded = cv2.imread('interior.jpg')
occluded = cv2.resize(occluded,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

mskBorder = cv2.imread('gradient_square.png')
mskBorder = cv2.resize(mskBorder,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

cannyFactor = 4.5
CANNY_MIN_LIMIT = 25
CANNY_MAX_LIMIT = 75
silhouetteBlend = 3.0

def computeLines(lines,img,upLine,downLine):

	if((not(type(lines)==np.ndarray)) and lines==None):
		return None

	step=0
	#print len(lines[0])
	if(len(lines)>30):
		step=int(len(lines)/30)
	else:
		step=1

	for i in range(0,len(lines)-1,step):

		rho,theta=lines[i][0]

		if((theta<5.5*np.pi/18)or(theta>12.5*np.pi/18)):
			continue

		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		yl=int(y0+(x0/b)*a)
		yr=int(y0-((HOUGH_WIDTH-x0)/b)*a)
		ratio=(HEIGHT/(HOUGH_HEIGHT*1.0))
		if((yl<HOUGH_HEIGHT/2)and(yr<HOUGH_HEIGHT/2)):
			if(yl<yr):
				upLine[0]=upLine[0]+rho
				upLine[1]=upLine[1]+theta
				upLine[2]=upLine[2]+1
				#cv2.line(img,(0,int(yl*ratio)),(WIDTH,int(yr*ratio)),(255,0,0),1)
			else:
				upLine[3]=upLine[3]+rho
				upLine[4]=upLine[4]+theta
				upLine[5]=upLine[5]+1
				#cv2.line(img,(0,int(yl*ratio)),(WIDTH,int(yr*ratio)),(30,190,50),1)

		if((yl>HOUGH_HEIGHT/2)and(yr>HOUGH_HEIGHT/2)):
			if(yl<yr):
				downLine[0]=downLine[0]+rho
				downLine[1]=downLine[1]+theta
				downLine[2]=downLine[2]+1
				#cv2.line(img,(0,int(yl*ratio)),(WIDTH,int(yr*ratio)),(190,50,30),1)
			else:
				downLine[3]=downLine[3]+rho
				downLine[4]=downLine[4]+theta
				downLine[5]=downLine[5]+1
				#cv2.line(img,(0,int(yl*ratio)),(WIDTH,int(yr*ratio)),(0,255,0),1)

	higherPt=((0,0),(WIDTH,0))
	lowerPt=((0,HEIGHT),(WIDTH,HEIGHT))

	rho=0
	theta=np.pi/2

	if(upLine[2]+upLine[5]>0):
		if(upLine[2]>upLine[5]):
			rho=upLine[0]/upLine[2]
			theta=upLine[1]/upLine[2]
		elif (upLine[5]>upLine[2]):
			rho=upLine[3]/upLine[5]
			theta=upLine[4]/upLine[5]
		else:
			#print "mean sup"
			rho=(upLine[0]+upLine[3])/(upLine[2]+upLine[5])
			theta=(upLine[1]+upLine[4])/(upLine[2]+upLine[5])

	newPerspPts[0]=rho
	newPerspPts[1]=theta

	rho=HOUGH_HEIGHT
	theta=np.pi/2

	if(downLine[2]+downLine[5]>0):
		if(downLine[2]>downLine[5]):
			rho=downLine[0]/downLine[2]
			theta=downLine[1]/downLine[2]
		elif (downLine[5]>downLine[2]):
			rho=downLine[3]/downLine[5]
			theta=downLine[4]/downLine[5]
		else:
			#print "mean inf"
			rho=(downLine[0]+downLine[3])/(downLine[2]+downLine[5])
			theta=(downLine[1]+downLine[4])/(downLine[2]+downLine[5])

	newPerspPts[2]=rho
	newPerspPts[3]=theta

	#Smooths perspective
	perspLines[0] = PERSPECTIVE_SMOOTH * perspLines[0] + (1 - PERSPECTIVE_SMOOTH) * newPerspPts[0]
	perspLines[1] = PERSPECTIVE_SMOOTH * perspLines[1] + (1 - PERSPECTIVE_SMOOTH) * newPerspPts[1];
	perspLines[2] = PERSPECTIVE_SMOOTH * perspLines[2] + (1 - PERSPECTIVE_SMOOTH) * newPerspPts[2];
	perspLines[3] = PERSPECTIVE_SMOOTH * perspLines[3] + (1 - PERSPECTIVE_SMOOTH) * newPerspPts[3];

	#Compute perspective lines
	#Superior line
	a = np.cos(perspLines[1])
	b = np.sin(perspLines[1])
	rho = perspLines[0] * (WIDTH / (HOUGH_WIDTH * 1.0));
	x0 = a*rho
	y0 = b*rho
	yl=int(y0+(x0/b)*a)
	yr=int(y0-((WIDTH-x0)/b)*a)
	if(yl<yr):
		higherPt=((0,0),(WIDTH,yr-yl))
		#cv2.line(img,(0,0),(WIDTH,yr-yl),(30,50,190),3)
	else:
		higherPt=((0,yl-yr),(WIDTH,0))
		#print higherPt
		#cv2.line(img,(0,yl-yr),(WIDTH,0),(30,50,190),3)

	#Inferior line
	a = np.cos(perspLines[3])
	b = np.sin(perspLines[3])
	rho = perspLines[2] * (WIDTH / (HOUGH_WIDTH * 1.0));
	x0 = a*rho
	y0 = b*rho
	yl=int(y0+(x0/b)*a)
	yr=int(y0-((WIDTH-x0)/b)*a)
	if(yl<yr):
		lowerPt=((0,HEIGHT-(yr-yl)),(WIDTH,HEIGHT))
		#cv2.line(img,(0,HEIGHT-(yr-yl)),(WIDTH,HEIGHT),(30,50,190),3)
	else:
		lowerPt=((0,HEIGHT),(WIDTH,HEIGHT-(yl-yr)))
		#cv2.line(img,(0,HEIGHT),(WIDTH,HEIGHT-(yl-yr)),(30,50,190),3)

	return higherPt+lowerPt


while(cap.isOpened()):

	ret, frame = cap.read()
	if(ret==False):
		break

	upLine=[0,0,0,0,0,0]
	downLine=[0,0,0,0,0,0]

	frame = cv2.resize(frame,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)

	gray = cv2.resize(frame,(HOUGH_WIDTH, HOUGH_HEIGHT), interpolation = cv2.INTER_CUBIC)
	gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

	cannyEdges = cv2.Canny(gray, CANNY_MIN_LIMIT * cannyFactor, CANNY_MAX_LIMIT * cannyFactor, apertureSize = 3)

	#EDGES
	kernel = np.ones((5,5),np.uint8)
	edges = cv2.dilate(cannyEdges,kernel,iterations = 2)
	edges = cv2.GaussianBlur(edges,(7,7),5)
	_,edges=cv2.threshold(edges,silhouetteBlend*255,0, cv2.THRESH_TRUNC)
	edges = cv2.bitwise_not(edges)
	edges = cv2.cvtColor(edges,cv2.COLOR_GRAY2RGB)
	edges = cv2.resize(edges,(WIDTH, HEIGHT), interpolation = cv2.INTER_CUBIC)


	#HOUGH LINES
	lines = cv2.HoughLines(cannyEdges,2,np.pi/180,100)
	ptsPersp = computeLines(lines,frame,upLine,downLine)
	#print ptsPersp
	pts1 = np.float32([[0,0],[WIDTH,0],[0,HEIGHT],[WIDTH,HEIGHT]])
	pts2 = np.float32([ptsPersp[0],ptsPersp[1],ptsPersp[2],ptsPersp[3]])
	perspTrans = cv2.getPerspectiveTransform(pts1,pts2)
	perspMskBorder = cv2.warpPerspective(mskBorder,perspTrans,(WIDTH,HEIGHT))
	perspMskCenter = cv2.bitwise_not(perspMskBorder)

	silhouette=cv2.subtract(frame,edges)
	silhouette=cv2.subtract(silhouette,perspMskCenter)

	occludedDisplay = cv2.subtract(occluded,perspMskCenter)
	occludedDisplay = cv2.subtract(occludedDisplay,silhouette)

	xRayView = cv2.subtract(frame,perspMskBorder)
	xRayView = cv2.add(xRayView,occludedDisplay)
	xRayView = cv2.add(xRayView,silhouette)

	cv2.imshow('Frame',xRayView)
	if cv2.waitKey(50) & 0xFF == ord('q'): #Regula a velocidade de exibição do vídeo
		break

cap.release()
cv2.waitKey()
cv2.destroyAllWindows()

	#SILHOUETTE
	# mask_inv = cv2.bitwise_not(perspMskCenter)
	# img2gray = cv2.cvtColor(mask_inv,cv2.COLOR_BGR2GRAY)
	# _,binMsk=cv2.threshold(img2gray,10,255, cv2.THRESH_BINARY)
	# mask_inv = cv2.bitwise_not(binMsk)
	# silhouette=cv2.subtract(frame,edges)
	# silhouette=cv2.bitwise_and(silhouette,silhouette,mask=binMsk)
	# # silhouette=cv2.subtract(silhouette,perspMskCenter)
	# xRayView=cv2.bitwise_and(frame,frame,mask=mask_inv)
	# #xRayView=cv2.subtract(frame,perspMskBorder)
	#
	# #virtView=cv2.bitwise_and(occluded,occluded,mask=binMsk)
	#
	# #edgesInv=cv2.bitwise_not(edges)
	#
	# virtView = cv2.subtract(occluded,perspMskCenter)
	# virtView = cv2.subtract(virtView,silhouette)
	#
	# xRayView=cv2.add(xRayView,virtView)
	# xRayView=cv2.add(xRayView,silhouette)
	# #xRayView=virtView
