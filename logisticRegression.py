import snap
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn import datasets
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
clf = LogisticRegression(C=1, penalty='l2',tol=0.001)
clf.fit(trainingExamplesX_scaled,trainingExamplesY)

#Here are the coefficients and intercept term
coef = clf.coef_
intercept = clf.intercept_

#Compute prediction accuracy
testPredictions = clf.predict(testingExamplesX_scaled)
numIncorrect = sum(testPredictions!=testingExamplesY)
numCorrect = sum(testPredictions==testingExamplesY)
print "Logistic Regression, Wikipedia Dataset\nPercent Correct: %.2f" % (100.0*float(numCorrect) / len(testingExamplesY))