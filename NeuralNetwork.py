import numpy as np


class NeuralNetwork(object):

    def __init__(self, input_size, *args):
        self.theta = np.ones((1, input_size))

    def test(self):
        print self.theta


if __name__ == '__main__':
    NeuralNetwork(500).test()
