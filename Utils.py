import numpy as np
import math

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def insert_bias(matrix):
    matrix = np.insert(matrix, 0, 1,
            axis=1)
    return matrix

def insert_bias_row(matrix):
    matrix = np.insert(matrix, 0, 1,
            axis=0)
    return matrix

def normalize(matrix):
    matrix_norm = matrix
    mu = np.zeros((1, matrix.shape[1]))
    sigma = np.zeros((1, matrix.shape[1]))

    mu = np.mean(matrix, axis=0)
    sigma = np.std(matrix, axis=0)

    matrix_norm = (matrix - mu) / sigma

    return matrix_norm

def sigmoid_gradient(matrix):
    return np.multiply(sigmoid(matrix),(1-sigmoid(matrix)))

def f(nn_params, *args):
    cls, X, y = args
    J, grad = cls.nn_cfx(X, y, nn_params)
    print J
    return J

def fprime(nn_params, *args):
    cls, X, y = args
    J, grad = cls.nn_cfx(X, y, nn_params)
    return grad

def wrap(*args):
    matrix = np.ones((1, 1))
    for a in args:
        matrix = np.append(matrix, a.reshape((1, a.size)))
    return matrix[1:]

def initialize_epsilon(L_in, L_out):
    return (math.sqrt(6)*1.0)/(math.sqrt(L_in+L_out+1))

def predict(Theta1, Theta2, X):
    m = X.shape[0]

    num_labels = Theta2.shape[0]
    p = np.zeros((m, 1))
    h1 = sigmoid(insert_bias(X).dot(Theta1.T))
    h2 = sigmoid(insert_bias(h1).dot(Theta2.T))

    p = h2.argmax(axis=1)

    print p

def accuracy(p, y):
    
