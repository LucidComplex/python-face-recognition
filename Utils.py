import numpy as np


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

def wrap(*args):
    matrix = np.zeros(args[0].shape)
    for a in args:
        matrix = np.append(matrix, a, axis=0)
