#Facial recognition using trained data

import numpy as np
import cv2
import os

#Changing the working directory
os.chdir("/home/pi/Face-Recognition/__pycache__/preview/OpenCV/opencv/data/haarcascades")


recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('/home/pi/Face-Recognition/training/training.yml')
cascadePath = "/home/pi/Face-Recognition/__pycache__/preview/OpenCV/opencv/data/haarcascades/haarcascade_frontalface_alt.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

#Initializing id counter
id = 0

names = ['None', 'Ruth','Barrack Obama', 'Uhuru Kenyatta', 'Mike', 'Trevor']

cam = cv2.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

#Define minimum window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        #minSize = (int(minW), int(minH)),
        )
    
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])
        
        #Checking cinfidence level
        if (confidence <100):
            id = names[id]
            confidence = "{0}%".format(round(100 - confidence))
        else:
            id = "intruder"
            confidence = "{0}%".format(round(100 - confidence))
            
        cv2.putText(img, str(id), (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

print("\n [INFO] Face Recognized. Exiting Program and cleanup")
cam.release()
cv2.destroyAllWindows()
