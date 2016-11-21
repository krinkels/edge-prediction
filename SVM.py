import snap
import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler

import featureextraction
import graphloading


#Load graph and training/testing examples
graph = graphloading.loadWikiGraph()
(trainingExamplesX, trainingExamplesY, testingExamplesX, testingExamplesY) = graphloading.generateExamples(graph)

#Scale features to have 0 mean and 1 variance
trainingExamplesX_scaled = StandardScaler().fit_transform(trainingExamplesX)
testingExamplesX_scaled = StandardScaler().fit_transform(testingExamplesX)

#Do logistic regression
clf = svm.SVC()
clf.fit(trainingExamplesX_scaled,trainingExamplesY)

#Compute prediction accuracy
testPredictions = clf.predict(testingExamplesX_scaled)
numIncorrect = sum(testPredictions!=testingExamplesY)
numCorrect = sum(testPredictions==testingExamplesY)
print "SVM with Radial Basis Function, Wikipedia Dataset\nPercent Correct: %.2f" % (100.0*float(numCorrect) / len(testingExamplesY))