import os
import json
import numpy as np
import cv2
import pyforms
from pyforms import BaseWidget
from pyforms.Controls import ControlText, ControlLabel, ControlButton, ControlImage, ControlFile, ControlDir

from FaceDetection import FaceDetection
from NeuralNetwork import NeuralNetwork


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
        self._predictbutton = self.__predictbAction

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

        self.nn = self.__init_nn()

    def __predictbAction(self):
        predictset_filename = 'predictset.csv'
        np.savetxt(predictset_filename,self.predictset, delimiter=',')

    def __init_nn(self):
        nn = NeuralNetwork()
        try:
            with open('nn_config.json') as nn_file:
                for config in nn_file:
                    config = json.loads(config)
                    nn = NeuralNetwork(config=config)
        except IOError:
            pass

        return nn

    def __change_path(self):
        image = cv2.imread(self._selectfile.value)
        self._predictimage.value = FaceDetection().drawrectangle(image)
        resizedimage = FaceDetection().resizeimageb(self._predictimage.value)
        croppedimage = FaceDetection().cropface(resizedimage)
        resizedcroppedimage = FaceDetection().resizeimagea(croppedimage)
        self.predictset = np.array(resizedcroppedimage).flatten()

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
        trainingset_filename = 'trainingset_' + self._pername.value + '.csv'
        np.savetxt(trainingset_filename, self.trainingset, delimiter=",")
        self.nn.train(trainingset_filename)

if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)
