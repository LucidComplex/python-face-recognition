import numpy as np
from scipy import optimize
from Utils import (sigmoid, sigmoid_gradient,
    insert_bias, insert_bias_row, normalize, f, fprime)

class NeuralNetwork(object):
    INIT_EPSILON = 0.12
    input_size = 20 * 20
    hidden_size = 25
    lambda_ = 1

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
        self.nn_cfx(X, y)

        res1 = optimize.fmin_cg(f, self.nn_params, fprime=fprime, args=self.nn_cfx(X, y), maxiter=50)
        print res1

    def test(self):
        self.train('who')
        pass


    def nn_cfx(self, X, y):
        nn_params = self.nn_params
        input_size = self.input_size
        num_labels = self.num_labels
        hidden_size = self.hidden_size
        lambda_ = self.lambda_
        
        theta1 = nn_params[:((hidden_size) * (input_size + 1))].reshape(
            (hidden_size, input_size + 1))
        
        theta2 = nn_params[((hidden_size) * (input_size + 1)):].reshape(
            (num_labels, hidden_size + 1))
        m = X.shape[0]

        J = 0
        theta1_grad = np.zeros(theta1.shape)
        theta2_grad = np.zeros(theta2.shape)

        a1 = insert_bias(X)

        z2 = theta1.dot(a1.T)
        a2 = sigmoid(z2)
        
        a2 = insert_bias(a2.T)

        z3 = theta2.dot(a2.T)
        h = sigmoid(z3)
        
        yk = np.zeros((num_labels, m))
        

        #back propagation

        for i in range(m):
            yk[int(y[i])-1, i] = 1.0

        error = (-yk) * np.log(h) - (1 - yk) * np.log(1 - h)
        J = (1.0/m)*sum(sum(error))

        t1 = np.array(theta1[:,1:])
        t2 = np.array(theta2[:,1:])

        sum1 = sum(sum(np.power(t1,2)))
        sum2 = sum(sum(np.power(t2,2)))

        r = (lambda_/(2.0*m))*(sum1 + sum2)
        J += r

        for t in range(m):
            z2 = np.matrix(theta1.dot(a1[t,:].T)).T #change to t later
            a2 = sigmoid(z2)
            a2 = insert_bias_row(a2)

            z3 = theta2.dot(a2)
            h = sigmoid(z3)

            z2 = insert_bias_row(z2)

            output = np.matrix(yk[:,t]).T #change to t later

            d3 = np.matrix(h - output)
            sg = np.matrix(sigmoid_gradient(z2))
            d2 = np.multiply(theta2.T.dot(d3),sg)
            d2 = d2[1:,:]

            theta2_grad += d3.dot(a2.T)
            theta1_grad += d2.dot(np.matrix(a1[t,:])) #change to t later

        # regularization

        theta1_grad[:,0] = np.matrix(theta1_grad[:,0]/(m*1.0))
        theta1_grad[:,1:] = (theta1_grad[:,1:]*(1/(m*1.0)) + ((lambda_/(m*1.0)*theta1[:,1:])))
        
        theta2_grad[:,0] = np.matrix(theta2_grad[:,0]/(m*1.0))        
        theta2_grad[:,1:] = (theta2_grad[:,1:]*(1/(m*1.0)) + ((lambda_/(m*1.0)*theta2[:,1:])))

        return J, np.append(theta1_grad.flatten(), theta2_grad.flatten())
        

if __name__ == '__main__':
    NeuralNetwork(10).test()
