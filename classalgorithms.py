from __future__ import division  # floating point division

import numpy as np

import utilities as utils


class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """

    def __init__(self, parameters={}):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}

    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params, parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}

    def getparams(self):
        return self.params

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """

    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest


class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """

    def __init__(self, parameters={}):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1

        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(
            np.add(np.dot(Xtrain.T, Xtrain) / numsamples, self.params['regwgt'] * np.identity(Xtrain.shape[1]))),
            Xtrain.T), yt) / numsamples

    def predict(self, Xtest):
        """
        :param Xtest: The testing data set with n samples and d features
        :return: ytest: array type, with n predictive value associate with the testing data
        """
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1
        ytest[ytest < 0] = 0
        return ytest

class LogitReg(Classifier):
    def __init__(self, parameters={}):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)
        self.weights = []
        # Weight Matrix
        self.C = None

    def learn(self, Xtrain, ytrain):
        # Initializing weight matrix
        self.C = self.params['regwgt'] * np.eye(Xtrain.shape[0])
        # Logistic regression learning function
        if self.params['regularizer'] == 'None':
            self.weights = np.dot(np.dot(np.linalg.pinv(np.dot(Xtrain.T, Xtrain)), Xtrain.T), ytrain)
            alpha = 0.01
            #Stochastic Method

            for t in range(Xtrain.shape[0]):
                xt = Xtrain[t, :]
                self.weights = self.weights + alpha * self.C[t,t] * (ytrain[t] - self.sigmoid((self.weights * xt).sum())) * xt

    def prob_vec(self, w, Xtrain):
        p = []
        numsample = Xtrain.shape[0]

        for i in range(numsample):
            p.append(self.sigmoid((Xtrain[i, :] * w).sum()))
        return np.array(p)

    def sigmoid(self, t):
        return 1.0 / (1 + np.exp(-t))

    def predict(self, Xtest):
        ytest = []
        for i in range(Xtest.shape[0]):
            if self.sigmoid((self.weights * Xtest[i, :]).sum()) >= 0.5:
                ytest.append(1.0)
            else:
                ytest.append(0.0)
        return np.array(ytest)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape, ))

            # TODO: implement learn and predict functions

