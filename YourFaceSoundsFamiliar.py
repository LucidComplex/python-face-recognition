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
        self._nametopred = ControlText('Name')
        self._selectfile = ControlFile()
        self._selectfile.changed = self.__change_path
        self._predictimage = ControlImage()
        self._predictbutton = ControlButton('Predict')
        self._predicteddetails = ControlLabel('Details')
        self._name = ControlLabel('Recognized Name: ')
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
            'Predict':['_selectfile','=','_nametopred','=','_predictimage',
                       '=','_predictbutton','=',
                       '_predicteddetails','=','_name',
                       '=','_fscore'],
            'Train': ['_pername','=','_selectdir',
                      '=','_imagetotrain','=','_addtolistbutton','=',
                      '_totrainlist','=','_trainbutton']
            }]
        self.trainingsetall = []
        self.nn = self.__init_nn()
        self.learned = {}



    def __predictbAction(self):
        predictset_filename = 'predictset.csv'
        np.savetxt(predictset_filename,self.predictset, delimiter=',')
        self.nn.predict(self.predictset)

    def __init_nn(self):
        nn = NeuralNetwork()
        return nn

    def __change_path(self):
        image = cv2.imread(self._selectfile.value)
        self._predictimage.value = []
        self._predictimage.value = FaceDetection().drawrectangle(image)
        resizedimage = FaceDetection().resizeimageb(self._predictimage.value)
        croppedimage = FaceDetection().cropface(resizedimage)
        resizedcroppedimage = FaceDetection().resizeimagea(croppedimage)
        self.predictset = np.array(resizedcroppedimage[1]).flatten()

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
        config = {'input_size': 30 * 30,  'hidden_size': 30 * 30, 'lambda': 1, 'num_labels': (len(self.learned))}
        self.nn = NeuralNetwork(config=config)
        X_matrix = np.empty((1, 1))
        y_matrix = np.empty((1, 1))
        m = 0
        for k, v in self.learned.iteritems():
            n = 0
            with open(k) as file_:
                all_lines = []
                for line in file_:
                    m += 1
                    n += 1
                    all_lines += line.split(',')
                X = np.array([all_lines], dtype=np.float)
            X = X.reshape((1, X.size))
            X_matrix = np.append(X_matrix, X)

            y = np.empty((n, 1))
            y.fill(v)
            y_matrix = np.append(y_matrix, y)
        X_matrix = X_matrix[1:]
        X_matrix = X_matrix.reshape((m, 30 * 30))
        y_matrix = y_matrix[1:]

        np.savetxt('X.csv', X_matrix, delimiter=',')
        np.savetxt('y.csv', y_matrix, delimiter=',')

        self.nn.train('X.csv', 'y.csv')

    def __addtolistbAction(self):
        trainingset_filename = 'trainingset_' + self._pername.value + '.csv'
        if trainingset_filename not in self.learned:
            self.learned[trainingset_filename] = len(self.learned) + 1
            self._totrainlist.__add__([self._pername.value])
            np.savetxt(trainingset_filename, self.trainingsetimage,
                delimiter=',')


if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)
