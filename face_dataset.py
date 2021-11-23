#Dataset for storing face features

import cv2
import os

#cam variable denotes video capture object
cam = cv2.VideoCapture(0)
cam.set(3, 640) #video width
cam.set(4, 480) #video height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#For each face captured
face_id = input('\n Enter user ID and press Enter:')
print('\n initializing face capture process. Face the camera and wait...')
 #Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    #Drawing rectangle around detected face with blue color
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        count += 1

        #Saving captured image in folder dataset
        cv2.imwrite("dataset/User." + str(face_id) + '.' +str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff #Press 'ESC' for exiting video
    if k ==27:
        break
    elif count >= 30:
        break

print("\n [INFO]Face captured. Exiting program and cleaning up")
cam.release()
cv2.destroyAllWindows()
