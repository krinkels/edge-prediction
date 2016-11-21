import numpy as np

from sklearn import svm
from sklearn.preprocessing import StandardScaler

def train(trainingExamplesX, trainingExamplesY):

    #Scale features to have 0 mean and 1 variance
    #Turns out this gives bad results since you normalize differently for different data sets
    trainingExamplesX_scaled = trainingExamplesX#StandardScaler().fit_transform(trainingExamplesX)

    #Do logistic regression
    clf = svm.SVC()
    clf.fit(trainingExamplesX_scaled,trainingExamplesY)
    
    return clf