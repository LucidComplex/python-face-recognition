import os
import json
import numpy as np
import cv2
import pyforms
from pyforms import BaseWidget
from pyforms.Controls import ControlText, ControlLabel, ControlButton, ControlImage, ControlFile, ControlDir, \
    ControlList

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
        self._predictbutton.value = self.__predictbAction

        #Train Tab
        self._pername = ControlText('Name')
        self._selectdir = ControlDir()
        self._selectdir.changed = self.__change_path_dir
        self._imagetotrain = ControlImage()
        self._totrainlist = ControlList("To Train",defaultValue=[])
        self.traininglist = self._totrainlist.value
        self._addtolistbutton = ControlButton('Add')
        self._addtolistbutton.value = self.__addtolistbAction
        self._trainbutton = ControlButton('Train')
        self._trainbutton.value = self.__trainbAction
        self._formset = [{
            'Predict':['_selectfile','=','_predictimage',
                       '=','_predictbutton','=',
                       '_predicteddetails','=','_name',
                       '=','_fscore'],
            'Train': ['_pername','=','_selectdir',
                      '=','_imagetotrain','=','_addtolistbutton','=',
                      '_totrainlist','=','_trainbutton']
            }]
        self.trainingsetall = []
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
        name = self._selectdir.value
        name = name.split('/')
        self._pername.value = name.pop(len(name)-1)
        self._imagetotrain.value = []
        listofimages = os.listdir(self._selectdir.value)
        listofimages = [cv2.imread(os.path.join(self._selectdir.value, filename)) for filename in listofimages]
        resizedimages = [FaceDetection().resizeimageb(image) for image in listofimages]
        croppedimages = [FaceDetection().cropface(image) for image in resizedimages]
        resized_images = [FaceDetection().resizeimagea(image) for image in croppedimages if image is not None]
        resizedcroppedimages = [image[0] for image in resized_images]
        resizedcroppedimagesgray = [image[1] for image in resized_images]
        self.trainingsetimage = [np.array(image).flatten() for image in resizedcroppedimagesgray]
        self._imagetotrain.value = resizedcroppedimages

    def __trainbAction(self):
        trainingset_filename = 'trainingset.csv'
        np.savetxt(trainingset_filename, self.trainingsetall, delimiter=",")
        self.nn.train(trainingset_filename)

    def __addtolistbAction(self):
        if self.traininglist.__contains__(self._pername.value):
            print('present')
        else:
            self.traininglist.append(self._pername.value)
            self.trainingsetall.append(self.trainingsetimage)
        self._totrainlist.value = self.traininglist
        print self.trainingsetall
if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)
