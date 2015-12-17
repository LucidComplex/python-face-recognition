import numpy as np
from Utils import sigmoid, insert_bias, normalize

class NeuralNetwork(object):
    INIT_EPSILON = 0.12
    input_size = 20 * 20
    hidden_size = 25
    lambda_ = 0

    def __init__(self, num_labels):
        self.num_labels = num_labels

        theta1 = np.zeros((1, self.input_size + 1))
        theta2 = np.zeros((1, self.hidden_size + 1))

        with open('Theta1.csv') as file_:
            all_lines = []
            for line in file_:
                all_lines += line.split(',')
            theta1 = np.array([all_lines], dtype=np.float)
        theta1 = theta1.reshape((self.hidden_size, self.input_size + 1))

        with open('Theta2.csv') as file_:
            all_lines = []
            for line in file_:
                all_lines += line.split(',')
            theta2 = np.array([all_lines], dtype=np.float)
        theta2 = theta2.reshape((self.num_labels, self.hidden_size + 1))
        
        #will use this later
        #theta1 = np.random.rand(self.hidden_size, self.input_size + 1)
        #theta2 = np.random.rand(self.num_labels, self.hidden_size + 1)
        self.nn_params = np.append(theta1.flatten(), theta2.flatten())

    def train(self, image):
        # X = image
        y = np.zeros((1, 1))
        m = 0
        with open('X.csv') as file_:
            all_lines = []
            for line in file_:
                m += 1
                all_lines += line.split(',')
            X = np.array([all_lines], dtype=np.float)
        X = X.reshape((m, self.input_size))
        with open('y.csv') as file_:
            all_lines = []
            for line in file_:
                all_lines += line.split(',')
            y = np.array([all_lines], dtype=np.float)
        y = y.reshape((m, 1))
        # X = normalize(X)
        self.cost_function(X, y)

    def test(self):
        self.train('who')
        pass


    def cost_function(self, X, y):
        nn_params = self.nn_params
        input_size = self.input_size
        num_labels = self.num_labels
        hidden_size = self.hidden_size
        
        theta1 = nn_params[:((hidden_size) * (input_size + 1))].reshape(
            (hidden_size, input_size + 1))
        
        theta2 = nn_params[((hidden_size) * (input_size + 1)):].reshape(
            (num_labels, hidden_size + 1))
        m = X.shape[0]

        J = 0
        theta1_grad = np.zeros(theta1.shape)
        theta2_grad = np.zeros(theta2.shape)

        X = insert_bias(X)

        z2 = theta1.dot(X.T)
        a2 = sigmoid(z2)
        
        a2 = insert_bias(a2.T)

        z3 = theta2.dot(a2.T)
        h = sigmoid(z3)
        
        yk = np.zeros((num_labels, m))
        
        for i in range(m):
            yk[int(y[i])-1, i] = 1.0

        error = (-yk) * np.log(h) - (1 - yk) * np.log(1 - h)
        J = (1.0/m)*sum(sum(error))

        

if __name__ == '__main__':
    NeuralNetwork(10).test()
