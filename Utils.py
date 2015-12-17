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
<<<<<<< HEAD

def wrap(*args):
    matrix = np.empty((1, 1))
    for a in args:
        print a.flatten()
        np.append(matrix, a.flatten(), axis=0)
    print matrix
=======
>>>>>>> 7034f5a5ec37147e31ceb441c6a3f494fd0d1219
