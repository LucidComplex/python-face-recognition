import numpy as np
import math
from fscore import fscore

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

def normalize(matrix, *args):
    matrix_norm = matrix
    if len(args) > 0:
        mu, sigma = args
    else:
        mu = np.mean(matrix, axis=0)
        sigma = np.std(matrix, axis=0)

    matrix_norm = (matrix - mu) / sigma

    return matrix_norm, mu, sigma

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

def predict1(Theta_1, Theta_2, X):
    m = X.shape[0]

    num_labels = Theta_2.shape[0]
    p = np.zeros((m, 1))
    a1 = insert_bias(X)
    h1 = sigmoid(a1.dot(Theta_1.T))
    
    z2 = insert_bias(h1)
    h2 = sigmoid(z2.dot(Theta_2.T))

    p = h2.argmax(axis = 1)

    for i in range(p.shape[0]):
        p[i] += 1
    return p

def accuracy(p, y):
    sum = 0.0
    m = p.shape[0]
    for i in range(m):
        if p[i] == y[i]:
            sum += 1
    accuracy = (sum/p.shape[0])*100
    return accuracy

#calculates total fscores of list of fscores
def total_fscore(*args):
    total = 0.0
    P_total = 0.0
    R_total = 0.0
    ctr = 0.0
    for a in args:
        for i in a:
            #print i.precision(), ', ', i.recall()
            total += i.calculate_f_score()
            ctr += 1
    return total/ctr

#outputs list of fscores for outputs
#p: already predicted value
#y: the output
#outputs: the number of outputs
def list_of_fscores(p, y, outputs):
    fscores = [fscore()]*int(outputs)
    for i in range(outputs):
        for j in range(y.shape[0]):
            #positive
            if p[j] == (i+1):
                #true
                if p[j] == y[j]:
                    fscores[i].true_pos += 1
                #false
                else:
                    fscores[i].false_pos += 1
            #negative
            elif p[j] != (i+1):
                #true
                if p[j] == y[j]:
                    fscores[i].true_neg += 1
                else:
                    fscores[i].false_neg += 1
    return fscores

if __name__ == '__main__':
    print total_fscore(list_of_fscores(np.array([1,1,1]), np.array([1,2,2]),2))
