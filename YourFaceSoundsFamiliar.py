import os
import numpy as np
import cv2
import pyforms
from pyforms import BaseWidget
from pyforms.Controls import ControlText, ControlLabel, ControlButton, ControlImage, ControlFile, ControlDir

from FaceDetection import FaceDetection


class YourFaceSoundsFamiliar(BaseWidget):
    def __init__(self):
        super(YourFaceSoundsFamiliar,self).__init__('Your Face Sounds Familiar')
        #Predict Tab
        self._imagepath = ControlText('Path')
        self._browsebuttonpredict = ControlButton('Browse')
        self._selectfile = ControlFile()
        self._selectfile.changed = self.__change_path
        self._predictimage = ControlImage()
        self._predictbutton = ControlButton('Predict')
        self._predicteddetails = ControlLabel('Details')
        self._name = ControlLabel('Name: ')
        self._fscore = ControlLabel('FScore: ')
        #Train Tab
        self._pername = ControlText('Name')
        self._selectdir = ControlDir()
        self._selectdir.changed = self.__change_path_dir
        self._imagetotrain = ControlImage()
        self._trainbutton = ControlButton('Train')
        self._trainbutton.value = self.__trainbAction
        self._formset = [{
            'Predict':['_selectfile','=','_predictimage',
                       '=','_predictbutton','=',
                       '_predicteddetails','=','_name',
                       '=','_fscore'],
            'Train': ['_pername','=','_selectdir',
                      '=','_imagetotrain','=','_trainbutton']
            }]


    def __change_path(self):
        image = cv2.imread(self._selectfile.value)
        self._predictimage.value = FaceDetection().drawrectangle(image)

    def __change_path_dir(self):
        self._imagetotrain.value = []
        listofimages = os.listdir(self._selectdir.value)
        listofimages = [cv2.imread(os.path.join(self._selectdir.value, filename)) for filename in listofimages]
        resizedimages = [FaceDetection().resizeimageb(image) for image in listofimages]
        croppedimages = [FaceDetection().cropface(image) for image in resizedimages]

        resized_images = [FaceDetection().resizeimagea(image) for image in croppedimages if image is not None]

        resizedcroppedimages = [image[0] for image in resized_images]
        resizedcroppedimagesgray = [image[1] for image in resized_images]
        self.trainingset = [np.array(image).flatten() for image in resizedcroppedimagesgray]
        self._imagetotrain.value = resizedcroppedimages

    def __trainbAction(self):
        np.savetxt("trainingset_"+ self._pername.value+".csv",self.trainingset,delimiter=",")

if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)