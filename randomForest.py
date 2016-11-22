from sklearn.ensemble import RandomForestClassifier

def train(trainingExamplesX, trainingExamplesY):

    #Do classification
    clf = RandomForestClassifier()
    clf.fit(trainingExamplesX,trainingExamplesY)
    
    return clf