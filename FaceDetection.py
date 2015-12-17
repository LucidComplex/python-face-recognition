import cv2
import sys
import numpy as np

class FaceDetection():
    def __init__(self):
        self._cascPath = "haarcascade_frontalface_default.xml"

    def  __detect_face(self, image):
        faceCascade = cv2.CascadeClassifier(self._cascPath)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
                    gray,
                    scaleFactor=1.2,
                    minNeighbors=5,
                    minSize=(30, 30),
                    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
                )
        return faces

    def cropface(self, image):
        faces = self.__detect_face(image)
        for (x, y, w, h) in faces:
            croppedface = image[y:y+h,x:x+w]
        return croppedface

    def drawrectangle(self, image):
        faces = self.__detect_face(image)
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        return image


if __name__ == '__main__':
    FaceDetection('/home/paks/Pictures/bang.jpg')