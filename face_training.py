# Training dataset using LBPH

import numpy as np
import cv2
from PIL import Image
import os

#Path of dataset
path = '/home/pi/Face-Recognition/dataset'

#os.chdir changes current working directory to a different path
os.chdir("/home/pi/Face-Recognition/__pycache__/preview/OpenCV/opencv/data/haarcascades")
recognizer = cv2.face.createLBPHFaceRecognizer()
detector = cv2.CascadeClassifier("/home/pi/Face-Recognition/__pycache__/preview/OpenCV/opencv/data/haarcascades/haarcascade_frontalface_alt.xml")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids
print("\n [INFO] Training faces. It will take a few seconds. Please wait...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

#Save the model into training_data.yml
recognizer.save('/home/pi/Face-Recognition/training/training.yml')

print("\ [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
