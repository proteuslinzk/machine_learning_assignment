from __future__ import division  # floating point division

import math

import numpy as np
import scipy.stats

import classalgorithms as algs
import dataloader as dtl

def kfold_split(Xtrain, ytrain, k, label):

    numsample = Xtrain.shape[0]
    subnumsample = int(math.floor(numsample / k))

    if label == k - 1:
        labelxtest = Xtrain[label * subnumsample: numsample, :]
        labelytest = ytrain[label * subnumsample: numsample]
        labelxtrain = Xtrain[0: label * subnumsample,:]
        labelytrain = ytrain[0: label * subnumsample]
    elif label == 0:
        labelxtest = Xtrain[0: subnumsample, :]
        labelytest = ytrain[0: subnumsample]
        labelxtrain = Xtrain[subnumsample: numsample, :]
        labelytrain = ytrain[subnumsample:numsample]
    else:
        labelxtest = Xtrain[label * subnumsample: (label + 1) * subnumsample, :]
        labelytest = ytrain[label * subnumsample: (label + 1) * subnumsample]
        labelxtrain = np.row_stack((Xtrain[0: label * subnumsample, :], Xtrain[(label + 1) * subnumsample : numsample, :]))
        labelytrain = np.concatenate((ytrain[0: label * subnumsample], ytrain[(label + 1) * subnumsample : numsample]))

    return (labelxtest, labelytest), (labelxtrain, labelytrain)

def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    randindices = np.random.randint(0, dataset.shape[0], trainsize + testsize)
    featureend = dataset.shape[1] - 1
    outputlocation = featureend
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0

    Xtrain = dataset[randindices[0:trainsize], featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize], outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize + testsize], featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize + testsize], outputlocation]

    if testdataset is not None:
        Xtest = dataset[:, featureoffset:featureend]
        ytest = dataset[:, outputlocation]

        # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility
    for ii in range(Xtrain.shape[1]):
        maxval = np.max(np.abs(Xtrain[:, ii]))
        if maxval > 0:
            Xtrain[:, ii] = np.divide(Xtrain[:, ii], maxval)
            Xtest[:, ii] = np.divide(Xtest[:, ii], maxval)

    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0], 1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0], 1))))

    return ((Xtrain, ytrain), (Xtest, ytest))


def getaccuracy(ytest, predictions):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == predictions[i]:
            correct += 1
    return (correct / float(len(ytest))) * 100.0


def geterror(ytest, predictions):
    return (100.0 - getaccuracy(ytest, predictions))


if __name__ == '__main__':
    trainsize = 5000
    testsize = 5000
    numruns = 3

    classalgs = {
         'Linear Regression': algs.LinearRegressionClass(),
        'Logistic Regression': algs.LogitReg(),
    }
    numalgs = len(classalgs)

    parameters = (
         {'regwgt': 0.0, 'nh': 4},
        {'regwgt': 0.01, 'nh': 8},
        {'regwgt': 0.05, 'nh': 16},
        {'regwgt': 0.1, 'nh': 32},
        {'regwgt': 0.2, 'nh': 32},
    )
    numparams = len(parameters)

    linear_bestparam = {}
    logistic_bestparam = {}

    errors = {}
    for learnername in classalgs:
        errors[learnername] = np.zeros((numparams, numruns))

    # select parameter
    for r in range(numruns):
        trainset, testset = dtl.load_susy(trainsize, testsize)

        # trainset, testset = dtl.load_susy_complete(trainsize,testsize)
        print('Running on train={0} and test={1} samples for run {2}').format(trainset[0].shape[0], testset[0].shape[0],
                                                                              r)

        k = 20

        for p in range(numparams):
            params = parameters[p]
            for learnername, learner in classalgs.iteritems():
                # Reset learner for new parameters
                learner.reset(params)
                #print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
                # Train model
                error = []
                for i in range(k):
                    (kxtest, kytest), (kxtrain, kytrain) = kfold_split(trainset[0], trainset[1], k, label=i)
                    learner.learn(kxtrain, kytrain)
                # Test model
                    predictions = learner.predict(kxtest)
                    error.append(geterror(kytest, predictions))
                #print 'Error for ' + learnername + ': ' + str(np.array(error).mean())
                errors[learnername][p, r] = np.array(error).mean()

    for learnername, learner in classalgs.iteritems():
        besterror = np.mean(errors[learnername][0, :])
        bestparams = 0
        for p in range(numparams):
            aveerror = np.mean(errors[learnername][p, :])
            if aveerror < besterror:
                besterror = aveerror
                bestparams = p

        # Extract best parameters
        learner.reset(parameters[bestparams])
        #print 'Best parameters for ' + learnername + ': ' + str(learner.getparams())
        # print 'Average error for ' + learnername + ': ' + str(besterror) + ' +- ' + str(
        #     1.96 * np.std(errors[learnername][bestparams, :]) / math.sqrt(numruns))

        if learnername == 'Linear Regression':
            linear_bestparam = learner.getparams()
        else:
            logistic_bestparam = learner.getparams()


    # Compare two algorithms
    error_linear = []
    error_logistic = []

    # logistic regression


    for epochs in range(20):
        trainset, testset = dtl.load_susy(trainsize, testsize)
        for learnername, learner in classalgs.iteritems():
            # Reset learner for new parameters
            if learnername == 'Linear Regression':
                learner.reset(linear_bestparam)
            else:
                learner.reset(logistic_bestparam)
            print 'Running learner = ' + learnername + ' on parameters ' + str(learner.getparams())
            # Train model
            learner.learn(trainset[0], trainset[1])
            # Test model
            predictions = learner.predict(testset[0])
            error = geterror(testset[1], predictions)
            print 'Error for ' + learnername + ': ' + str(error)
            if learnername == 'Linear Regression':
                error_linear.append(error)
            else:
                error_logistic.append(error)


    # t-test
    t, pvalue = scipy.stats.ttest_ind(error_linear, error_logistic)
    print 'pvalue is :'
    print pvalue

    if pvalue < .05:
        Hypo = False
    else:
        Hypo = True

    if not Hypo:
        print "Pvalue of the t-test is smaller than 0.05, so difference between the mean value of logistic regression"
        print "and the linear regression is significant."
        if np.mean(error_logistic) < np.mean(error_linear):
            print "Logistic perform better than linear regression."
        else:
            print "Linear regression perform better than logistic regression."

