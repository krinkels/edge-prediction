import numpy as np
import logisticRegression
import neuralNetwork
import SVM
import sys


def edgePrediction(classifier='LR',trainFilename = 'Wiki',testFilenames = ['Astrophysics']):
    """
    This function trains a classifier on a training data set and computes the testing accuracy for other data sets.
    
    Args:
        classifier: A string specifying which classifier to use. Currently, must either be 'LR','SVM', or 'NN'
        trainFileName: String specifying which data set should be used to train the classifier
        testFileNames: Array of filenames specifying the data sets on which to make test predictions.
    """
    try:
        features = np.genfromtxt('featureFiles/' + trainFilename + '.features', delimiter=' ')
        labels   = np.genfromtxt('featureFiles/' + trainFilename + '.labels',   delimiter=' ')
    except IOError as e:
        print "Could not open featureFiles/" + trainFilename + ".features " + "or featureFiles/" + trainFilename + ".labels"
        return
    
    #Get training data
    splitIndex = np.argmin(labels)
    trainingExamplesX = features[:splitIndex]
    trainingExamplesY = labels[:splitIndex]
    print "Training with " + trainFilename + " Dataset:"
    
    #Select classifier and run classification
    if classifier.lower() == 'lr':
        print "Using logistic regression"
        sys.stdout.flush()
        clf = logisticRegression.train(trainingExamplesX,trainingExamplesY)
    elif classifier.lower() == 'svm':
        print "Using SVM"
        sys.stdout.flush()
        clf = SVM.train(trainingExamplesX,trainingExamplesY)
    elif classifier.lower() == 'nn':
        print "Using neural network"
        sys.stdout.flush()
        clf = neuralNetwork.train(trainingExamplesX,trainingExamplesY)
    else:
        print "Could not understand classifier: "  + classifier
        print "Using logistic regression"
        sys.stdout.flush()
        clf = logisticRegression.train(trainingExamplesX,trainingExamplesY)

    #Load testing data sets
    for filename in testFilenames:
        try:
            features = np.genfromtxt('featureFiles/' + filename + '.features', delimiter=' ')
            labels   = np.genfromtxt('featureFiles/' + filename + '.labels',   delimiter=' ')
        except IOError as e:
            print "Could not open featureFiles/" + filename + ".features " + "or featureFiles/" + filename + ".labels"
            sys.stdout.flush()
            continue
        
        splitIndex = np.argmin(labels)
        testingExamplesX = features[(splitIndex+1):]
        testingExamplesY = labels[(splitIndex+1):]
        
        #Compute prediction accuracy
        testPredictions = clf.predict(testingExamplesX)
        numCorrect = sum(testPredictions==testingExamplesY)
        print filename + ": Percent Correct: %.3f" % (100.0*float(numCorrect) / len(testingExamplesY))
        sys.stdout.flush()
    print " "
        
#This loops through all possible learners, training data sets, and testing data sets. I ran this and saved the print output for reference
filenames = ["Wiki","AstroPhysics","CondensedMatter","GeneralRelativity",'HEPhysics','HEPhysicsTheory']
for reg in ['lr','svm','nn']:
    for fn in filenames:
        edgePrediction(reg,fn,filenames)