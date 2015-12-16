import pyforms
from pyforms import BaseWidget
from pyforms.Controls import ControlText, ControlLabel, ControlButton, ControlImage

class YourFaceSoundsFamiliar(BaseWidget):
    def __init__(self):
        super(YourFaceSoundsFamiliar,self).__init__('Your Face Sounds Familiar')
        self._imagepath = ControlText('Path')
        self._browsebutton = ControlButton('Browse')
        self._predictimage = ControlImage()
        self._predictbutton = ControlButton('Predict')
        self._predicteddetails = ControlLabel('Details')
        self._name = ControlLabel('Name: ')
        self._fscore = ControlLabel('FScore: ')
        self._formset = [ {
            'Predict':['_imagepath','||','_browsebutton','=','_predictimage'
                       ,'=','_predictbutton','=',
                       '_predicteddetails','=','_name',
                       '=','_fscore'],
            'Train': ['']
            } ]


if __name__ == '__main__':
    pyforms.startApp(YourFaceSoundsFamiliar)