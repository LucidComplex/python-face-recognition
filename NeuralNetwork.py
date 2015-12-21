import sys
import numpy as np
from scipy import optimize
from Utils import (sigmoid, sigmoid_gradient, accuracy, predict1, list_of_fscores,
    insert_bias, insert_bias_row, normalize, wrap, f, fprime, initialize_epsilon,
    total_fscore)

class NeuralNetwork(object):

    def __init__(self, **kwargs):
        self.config = kwargs.get('config',
                {'input_size': 30 * 30, 'hidden_size': 30 * 30, 'lambda': 1,
                    'num_labels': 1})
        self.INIT_EPSILON = initialize_epsilon(self.config['input_size'],
            self.config['hidden_size'])
        input_size = self.config['input_size']
        hidden_size = self.config['hidden_size']
        num_labels = self.config['num_labels']
        try:
            theta1 = None
            theta2 = None
            with open('Theta1.csv') as theta1_file:
                all_lines = []
                for line in theta1_file:
                    all_lines += line.split(',')
                theta1 = np.array([all_lines], dtype=np.float)
            theta1 = theta1.reshape((hidden_size, input_size + 1))
            with open('Theta2.csv') as theta2_file:
                all_lines = []
                for line in theta2_file:
                    all_lines += line.split(',')
                theta2 = np.array([all_lines], dtype=np.float)
            theta2 = theta2.reshape((num_labels, hidden_size + 1))
        except (IOError, ValueError):
            theta1 = np.random.rand(hidden_size, input_size + 1) * 2 * self.INIT_EPSILON - self.INIT_EPSILON
            theta2 = np.random.rand(num_labels, hidden_size + 1) * 2 * self.INIT_EPSILON - self.INIT_EPSILON
        finally:
            self.nn_params = wrap(theta1, theta2)


    def clear(self):
        self.nn_params = None
        input_size = self.config['input_size']
        hidden_size = self.config['hidden_size']
        num_labels = self.config['num_labels']
        theta1 = np.random.rand(hidden_size, input_size + 1) * 2 * self.INIT_EPSILON - self.INIT_EPSILON
        theta2 = np.random.rand(num_labels, hidden_size + 1) * 2 * self.INIT_EPSILON - self.INIT_EPSILON
        self.nn_params = wrap(theta1, theta2)

    def train(self, X, y, cv, test_set, cv_y, test_y):
        self.clear()
        col = self.config['input_size']
        cv = np.reshape(cv, (len(cv)/col, col))
        test_set = np.reshape(test_set, (len(test_set)/col, col))

        m = 0

        X, mu, sigma = normalize(X)

        # save X, mu and sigma
        np.savetxt('X.csv', X, delimiter=',')
        np.savetxt('mu.csv', mu, delimiter=',')
        np.savetxt('sigma.csv', sigma, delimiter=',')

        self.nn_params = optimize.fmin_cg(f, self.nn_params, args=(self, X, y), maxiter=50,
            fprime=fprime)

        # save nn_parameters
        hidden_size = self.config['hidden_size']
        input_size = self.config['input_size']
        num_labels = self.config['num_labels']
        theta1 = self.nn_params[:((hidden_size) * (input_size + 1))].reshape(
            (hidden_size, input_size + 1))
        theta2 = self.nn_params[((hidden_size) * (input_size + 1)):].reshape(
            (num_labels, hidden_size + 1))
        np.savetxt('Theta1.csv', theta1, delimiter=',')
        np.savetxt('Theta2.csv', theta2, delimiter=',')

        #test the model
        p = predict1(theta1, theta2, X)
        _accuracy = accuracy(p, y)
        l_fscores = list_of_fscores(p, y, num_labels)
        fscore = total_fscore(l_fscores)

        #test the model on cross validation set
        fscores_cv = 0.0
        accuracy_cv = 0.0
        l_fscores_cv = 0.0
        no = 0

        X1, mu1, sigma1 = normalize(np.array(cv), mu, sigma)
        p_cv = predict1(theta1, theta2, X1)
        accuracy_cv = accuracy(p_cv, cv_y)
        l_fscores_cv = list_of_fscores(p_cv, cv_y, num_labels)
        fscores_cv = total_fscore(l_fscores_cv)

        #test the model on test set
        X_test, mu_test, sigma_test = normalize(np.array(test_set), mu, sigma)
        p_test = predict1(theta1, theta2, X_test)
        accuracy_test = accuracy(p_test, test_y)
        l_fscores_test = list_of_fscores(p_test, test_y, num_labels)
        fscore_test = total_fscore(l_fscores_test)

        print 'here are the shizz results:'
        print 'fscores in cross validation:'
        print fscores_cv, ' ', accuracy_cv
        print 'fscores in test set: ', fscore_test
        print 'accuracy_test: ', accuracy_test

        return self.nn_cfx(X, y, self.nn_params), self.nn_params, fscores_cv, fscore_test


    def predict(self, image_matrix):
        X = image_matrix.reshape((1, image_matrix.size))
        mu = np.loadtxt('mu.csv')
        sigma = np.loadtxt('sigma.csv')
        X, mu, sigma = normalize(X, mu, sigma)
        hidden_size = self.config['hidden_size']
        input_size = self.config['input_size']
        num_labels = self.config['num_labels']
        theta1 = self.nn_params[:((hidden_size) * (input_size + 1))].reshape(
            (hidden_size, input_size + 1))
        theta2 = self.nn_params[((hidden_size) * (input_size + 1)):].reshape(
            (num_labels, hidden_size + 1))

        a1 = insert_bias(X)

        z2 = theta1.dot(a1.T)
        a2 = sigmoid(z2)

        a2 = insert_bias(a2.T)
        print theta2.shape

        z3 = theta2.dot(a2.T)
        h = sigmoid(z3)
        print h
        return h

    def nn_cfx(self, X, y, nn_params):
        input_size = self.config['input_size']
        num_labels = self.config['num_labels']
        hidden_size = self.config['hidden_size']
        lambda_ = self.config['lambda']

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
        #print accuracy(predict1(theta1_grad, theta2_grad, X), y)
        return J, wrap(theta1_grad, theta2_grad)


if __name__ == '__main__':
    NeuralNetwork().train('trainingset_Gregory House.csv')
