import snap
import random
import featureextraction
import numpy as np

def loadWikiGraph():
    """
    Loads the Wikipedia vote graph

    Returns:
        A directed graph (snap.TNGraph) of the Wikipedia admin votes
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/wiki-Vote.txt", 0, 1, "\t")

def generateExamples(graph, testProportion=0.1, seed=0):
    """
    Generates testing and training examples for node pairs from the given graph

    Args:
        graph (snap.TNGraph or snap.TUNGraph) : the graph to generate examples from
        testingProportion (float) : the proportion of edges to use in the testing set
        seed (hashable) : a seed for the random number generator

    Returns:
        A tuple of training and testing examples of the form
        ( [ ([train_features_1], train_class_1),
            ([train_features_2], train_class_2), ...],
          [ ([test_features_1], test_class_1),
            ([test_features_2], test_class_2), ..]
        )
        where the classification is 1 if an edge exists and 0 if not
    """
    random.seed(seed)

    trainingExamples = []
    testingExamples = []

    # First, partition existing edges in training and testing sets
    splitIndex = graph.GetEdges() - int(graph.GetEdges() * testProportion)

    examplesX = [ featureextraction.extractFeatures(graph, edge.GetSrcNId(), edge.GetDstNId()) for edge in graph.Edges() ]
    examplesY = np.ones(int(graph.GetEdges()))

    random.shuffle(examplesX)
    trainingExamplesX = examplesX[:splitIndex]
    testingExamplesX = examplesX[splitIndex:]

    # Next, generate pairs of nodes which are not edges to balance out the training and testing sets
    nodeIDs = [ node.GetId() for node in graph.Nodes() ]
    nodePairs = set()
    while len(nodePairs) < graph.GetEdges():
        pair = tuple(random.sample(nodeIDs, 2))
        if not graph.IsEdge(pair[0], pair[1]):
            nodePairs.add(pair)

    nonEdgeExamplesX = [ featureextraction.extractFeatures(graph, pair[0], pair[1]) for pair in nodePairs ]
    nonEdgeExamplesY = np.zeros(int(graph.GetEdges()))

    trainingExamplesX += nonEdgeExamplesX[:splitIndex]
    testingExamplesX  += nonEdgeExamplesX[splitIndex:]
    trainingExamplesY =  np.concatenate((examplesY[:splitIndex],nonEdgeExamplesY[:splitIndex]), axis = 0)
    testingExamplesY  =  np.concatenate((examplesY[splitIndex:],nonEdgeExamplesY[splitIndex:]), axis = 0)

    return (trainingExamplesX,trainingExamplesY, testingExamplesX, testingExamplesY)
