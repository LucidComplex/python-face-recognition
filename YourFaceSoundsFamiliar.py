import pyforms
from pyforms import BaseWidget
from pyforms.Controls import ControlText, ControlLabel, ControlButton, ControlImage

class YourFaceSoundsFamiliar(BaseWidget):
    def __init__(self):
        super(YourFaceSoundsFamiliar,self).__init__('Your Face Sounds Familiar')
        #Predict Tab
        self._imagepath = ControlText('Path')
        self._browsebuttonpredict = ControlButton('Browse')
        self._predictimage = ControlImage()
        self._predictbutton = ControlButton('Predict')
        self._predicteddetails = ControlLabel('Details')
        self._name = ControlLabel('Name: ')
        self._fscore = ControlLabel('FScore: ')
        #Train Tab
        self._pername = ControlText('Name')
        self._browsebuttontrain = ControlButton('Browse')
        self._dirpath = ControlText('Path')
        self._imagetotrain = ControlImage()
        self._trainbutton = ControlButton('Train')
        self._formset = [ {
            'Predict':['_imagepath','||','_browsebuttonpredict','=','_predictimage',
                       '=','_predictbutton','=',
                       '_predicteddetails','=','_name',
                       '=','_fscore'],
            'Train': ['_pername','=','_dirpath','||','_browsebuttontrain',
                      '=','_imagetotrain','=','_trainbutton']
            } ]


if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)