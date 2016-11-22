from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

def train(trainingExamplesX, trainingExamplesY):

    #Scale features to have 0 mean and 1 variance
    #Turns out this gives bad results since you normalize differently for different data sets
    trainingExamplesX_scaled = trainingExamplesX#StandardScaler().fit_transform(trainingExamplesX)

    #Do classification
    clf = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30,30,30,30), random_state=1)
    clf.fit(trainingExamplesX_scaled,trainingExamplesY)
    
    return clf