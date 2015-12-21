import os
import json
import numpy as np
import cv2
import pyforms
from pyforms import BaseWidget
from pyforms.Controls import ControlText, ControlLabel, ControlButton, ControlImage, ControlFile, ControlDir, \
    ControlList

from operator import itemgetter
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
        # self._imagetotest = ControlImage()
        self._totrainlist = ControlList("To Train",defaultValue=[])
        self.traininglist = self._totrainlist.value
        self._addtolistbutton = ControlButton('Add')
        self._addtolistbutton.value = self.__addtolistbAction
        self._trainbutton = ControlButton('Train')
        self._trainbutton.value = self.__trainbAction

        #Formsets
        self._formset = [{
            'Predict':['_selectfile','=','_nametopred','=','_predictimage',
                       '=','_predictbutton','=',
                       '_predicteddetails','=','_name',
                       '=','_fscore'],
            'Train': ['_pername', '=', '_selectdir',
                      '=', '_imagetotrain', '=', '_addtolistbutton','=' ,
                      '_totrainlist', '=', '_trainbutton']
            }]
        self.trainingsetall = []
        self.nn = self.__init_nn()
        self.learned = {}
        self._k = 4
        self._trainingPercent = 0.8
        self.learned = self.__load_learned()
        self.cross_validation_set = [np.empty((0,0))]*self._k
        self.cross_validation_set_y = [np.empty((0,0))]*self._k
        self.test_set = np.empty((0, 0))
        self.testing_y = np.empty((0, 0))
        self.training_X = [np.empty((0, 900))] * self._k
        self.training_y = [np.empty((0, 1))] * self._k

        self.X = np.empty((0, 0))

    def __load_learned(self):
        try:
            with open('learned.json') as learned_file:
                for line in learned_file:
                    learned = json.loads(line)
                    for key in learned.keys():
                        self._totrainlist.__add__([key])
        except IOError:
            learned = {}

        config = {'input_size': 30 * 30,  'hidden_size': 30 * 30, 'lambda': 1, 'num_labels': (len(learned))}
        self.nn = NeuralNetwork(config=config)

        return learned

    def __predictbAction(self):
        predictset_filename = 'predictset.csv'
        np.savetxt(predictset_filename,self.predictset, delimiter=',')
        prediction = np.argmax(self.nn.predict(self.predictset)) + 1
        for k, v in self.learned.iteritems():
            if prediction == v:
                self._name.value = k



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
        # self._imagetotest.value = []
        listofimages = os.listdir(self._selectdir.value)
        listofimages = sorted(listofimages)
        listofimages = [cv2.imread(os.path.join(self._selectdir.value, filename)) for filename in listofimages]
        resizedimages = [FaceDetection().resizeimageb(image) for image in listofimages]
        croppedimages = [FaceDetection().cropface(image) for image in resizedimages]
        resized_images = [FaceDetection().resizeimagea(image) for image in croppedimages if image is not None]
        resizedcroppedimages = [image[0] for image in resized_images]
        resizedcroppedimagesgray = [image[1] for image in resized_images]
        trainthisImages = resizedcroppedimagesgray[0:int(len(resizedcroppedimagesgray)*self._trainingPercent)]
        testthisImages = resizedcroppedimagesgray[int(len(resizedcroppedimagesgray)*self._trainingPercent):]
        self.trainingsetimage = [np.array(image).flatten() for image in trainthisImages]
        self.testingsetimage = [np.array(image).flatten() for image in testthisImages]
        self._imagetotrain.value = trainthisImages

        self.cross_validation = [np.empty((0,0))]*self._k
        self.cv_size = [0]*self._k
        l = 0
        for j in self.trainingsetimage:
            if l == self._k:
                l = 0
            self.cross_validation[l] = np.append(self.cross_validation[l], j)
            self.cv_size[l] += 1
            l += 1
        self._imagetotrain.value = resizedcroppedimages


    def __trainbAction(self):
        config = {'input_size': 30 * 30,  'hidden_size': 30 * 30, 'lambda': 1, 'num_labels': (len(self.learned))}
        self.nn = NeuralNetwork(config=config)

        cost_params_fscore = []
        for i in range(self._k):
            cost_params_fscore.append(self.nn.train(self.training_X[i], self.training_y[i], self.cross_validation_set[i], self.test_set, self.cross_validation_set_y[i], self.testing_y))

        best_model = max(cost_params_fscore, key=itemgetter(2))
        print best_model[0], best_model[2]


    def __addtolistbAction(self):
        ### TEST ###
        if self._pername.value in self.learned:
            return
        X = np.array(self.trainingsetimage) # size: 8
        X_set = np.empty((0, 0)) # this will be a k-sized set of X slices
        slice_size = X.shape[0] / self._k # 2
        if self._pername.value not in self.learned:
            label = len(self.learned) + 1
            self.learned[self._pername.value] = label
        else:
            return
        for i in range(self._k):
            sliced = np.delete(X, slice(i*2, (i*2)+2), axis=0)
            self.training_X[i] = np.append(self.training_X[i], sliced, axis=0)
            y = np.empty((sliced.shape[0], 1))
            y.fill(label)
            self.training_y[i] = np.append(self.training_y[i], y, axis=0)
            print self.training_X[i].shape
        self._totrainlist.__add__([self._pername.value])

        with open('learned.json', 'w') as learned_file:
            learned_file.write(json.dumps(self.learned))
        ############


        for i in range(self._k):
            self.cross_validation_set[i]  = np.append(self.cross_validation_set[i], self.cross_validation[i])
            self.cross_validation_set_y[i] = np.append(self.cross_validation_set_y[i], [label]*self.cv_size[i])

        self.test_set = np.append(self.test_set, self.testingsetimage)
        self.testing_y = np.append(self.testing_y, [label]*(len(self.testingsetimage)))


if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)
