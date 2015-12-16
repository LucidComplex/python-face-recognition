import numpy as np


class NeuralNetwork(object):

    def __init__(self, input_size, output_size):
        self.hidden_layer_size = input_size / 4
        self.theta = np.ones((1, input_size))

    def train(self, image):


    def test(self):
        print self.theta


if __name__ == '__main__':
    NeuralNetwork(500, 10).test()
