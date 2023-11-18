import numpy as np
import cv2
import os
import dlib

detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def empty(a):
    pass
cv2.namedWindow('BRG')
cv2.resizeWindow('BRG',600,500)
cv2.createTrackbar("BLUE","BRG",0,255,empty)
cv2.createTrackbar("RED","BRG",0,255,empty)
cv2.createTrackbar("GREEN","BRG",0,255,empty)
def createBox(img,points):
    mask=np.zeros_like(img)
    mask=cv2.fillPoly(mask,[points],(255,255,255))
    return mask 
while True:
    img=cv2.imread('image/anh1.jpg')
    img=cv2.risize(img,(0,0),None,0.5,0.5)
    imgOriginal=img.copy()
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(imgGray)
    for face in faces:
        landmarks=predictor(imgGray,face)
        myPoints=[]
        for i in range(68):
            x=landmarks.part(i).x
            y=landmarks.part(i).y
            myPoints.append([x,y])
        myPoints=np.array(myPoints)
        imgLips=createBox(imgOriginal,myPoints[1:68])
        imgColorLips=np.zeros_like(imgLips)

        b=cv2.getTrackbarPos("BLUE","BRG")
        r=cv2.getTrackbarPos("RED","BRG")
        g=cv2.getTrackbarPos("GREEN","BRG")

        imgColorLips[:]=b,g,r
        imgColorLips=cv2.bitwise_and(imgLips,imgColorLips)
        imgColorLips=cv2.GaussianBlur(imgColorLips,(7,7),10)
        imgColorLips=cv2.addWeighted(imgOriginal,1,imgColorLips,0.5,0)
    cv2.imshow("IMG",imgColorLips)
    if cv2.waitKey(1)==ord("q"):
        break
        
