import snap
import random
import featureextraction
import numpy as np
import sys

#List of all filenames, which appear in the load_() functions
allFileNames = ['Wiki','AstroPhysics','CondensedMatter','GeneralRelativity','HEPhysics','HEPhysicsTheory']

def loadWiki():
    """
    Loads the Wikipedia vote graph

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/wiki-Vote.txt", 0, 1, "\t")

def loadAstroPhysics():
    """
    Loads the Arxiv Astro Physics collaboration network

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/ca-AstroPh.txt", 0, 1, "\t")

def loadCondensedMatter():
    """
    Loads the Arxiv Condensed Matter collaboration network

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/ca-CondMat.txt", 0, 1, "\t")

def loadGeneralRelativity():
    """
    Loads the Arxiv General Relativity collaboration network

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/ca-GrQc.txt", 0, 1, "\t")

def loadHEPhysics():
    """
    Loads the Arxiv High Energy Physics collaboration network

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/ca-HepPh.txt", 0, 1, "\t")

def loadHEPhysicsTheory():
    """
    Loads the Arxiv High Energy Physics Theory  collaboration network

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/ca-HepTh.txt", 0, 1, "\t")

def loadSlashdotOld():
    """
    Loads the Slashdot Zoo network from November 2008

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/soc-Slashdot0811.txt")

def loadSlashdotNew():
    """
    Loads the Slashdot Zoo network from February 2009

    Returns:
        A directed graph (snap.TNGraph)
    """
    return snap.LoadEdgeList(snap.PNGraph, "datasets/soc-Slashdot0902.txt")

def generateSlashdotExamples(seed=0, filename=None):
  
    # Load in the graphs. For the new graph, we only care about nodes which exist
    # in the old graph, so generate the subgraph first
    oldGraph = loadSlashdotOld()
    newGraph = loadSlashdotNew()

    oldNodeIDV = snap.TIntV()
    for node in oldGraph.Nodes():
        oldNodeIDV.Add(node.GetId())
    newGraph = snap.GetSubGraph(newGraph, oldNodeIDV)

    neighborTable = {}
    for node in graph.Nodes():
        neighbors = set([ nodeID for nodeID in node.GetOutEdges() ]) | set([ nodeID for nodeID in node.GetInEdges()])
        neighborTable[node.GetId()] = len(neighbors)

    # Get edge examples
    trainingExamplesX = [ featureextraction.extractFeatures(oldGraph, edge.GetSrcNId(), edge.GetDstNId(), neighborTable) for edge in oldGraph.Edges() ]
    trainingExamplesY = [1] * len(trainingExamplesX)

    testingExamplesX = [ featureextraction.extractFeatures(oldGraph, edge.GetSrcNId(), edge.GetDstNId(), neighborTable) for edge in newGraph.Edges()
                         if not oldGraph.IsEdge(edge.GetSrcNId(), edge.GetDstNId()) ]
    testingExamplesY = [1] * len(testingExamplesX)

    # Next, generate pairs of nodes which are not edges to balance out the training and testing sets
    # For the training set, we randomly sample pairs of nodes which do not form an edge in the old graph.
    # For the testing set, we randomly sample pairs of nodes which do not form an edge in the new graph.
    nodeIDs = [ node.GetId() for node in oldGraph.Nodes() ]
    trainingNodePairs = []
    testingNodePairs = []

    for srcID in nodeIDs:
        srcNode = oldGraph.GetNI(srcID)
        srcOutNeighbors = frozenset([ nodeID for nodeID in srcNode.GetOutEdges() ])
        srcInNeighbors = frozenset([ nodeID for nodeID in srcNode.GetInEdges() ])
        srcNeighbors = srcOutNeighbors | srcInNeighbors

        secondDegrees = set()
        for neighbor in srcNeighbors:
            neighborNode = oldGraph.GetNI(neighbor)
            nOutNeighbors = frozenset([ nodeID for nodeID in neighborNode.GetOutEdges() ])
            nInNeighbors = frozenset([ nodeID for nodeID in neighborNode.GetInEdges() ])
            nNeighbors = nOutNeighbors | nInNeighbors
            secondDegrees = secondDegrees | nNeighbors
        secondDegrees = secondDegrees - set([srcID])

        for dstID in secondDegrees:
            if not oldGraph.IsEdge(srcID, dstID):
                trainingNodePairs.append((srcID, dstID))
            if not newGraph.IsEdge(srcID, dstID):
                testingNodePairs.append((srcID, dstID))

    trainingNodePairs = random.sample(trainingNodePairs, len(trainingExamplesX))
    testingNodePairs = random.sample(testingNodePairs, len(testingExamplesX))

    trainingExamplesX += [ featureextraction.extractFeatures(oldGraph, pair[0], pair[1], neighborTable) for pair in trainingNodePairs ]
    testingExamplesX  += [ featureextraction.extractFeatures(oldGraph, pair[0], pair[1], neighborTable) for pair in testingNodePairs ]
    trainingExamplesY += [0] * len(trainingNodePairs)
    testingExamplesY  =  [0] * len(testingNodePairs)

    #Instead of empty space between training and testing sets, write -1's.
    #This is used as a flag to easily find separation between data sets
    #while using np.genfromtext()
    (length,numFeatures) = np.shape(trainingExamplesX)
    separator = np.zeros(numFeatures) - 1

    # Write the examples to a file if necessary
    #Note that the features are written to a .features file, and the
    #labels are written to a .labels file
    if filename is not None:
        try:
            f = open(filename + '.features', 'w')
            for example in trainingExamplesX:
                f.write(" ".join([ str(val) for val in example]))
                f.write("\n")
            f.write(" ".join([ str(val) for val in separator]))
            f.write("\n")
            for example in testingExamplesX:
                f.write(" ".join([ str(val) for val in example ]))
                f.write("\n")
            f.close()

            f = open(filename + '.labels', 'w')
            for example in trainingExamplesY:
                f.write(str(example))
                f.write("\n")
            f.write("-1\n")
            for example in testingExamplesY:
                f.write(str(example))
                f.write("\n")
            f.close()

        except IOError as e:
            print "Error writing to file {0}: {1}".format(filename, e.strerror)
        
    return (trainingExamplesX,trainingExamplesY, testingExamplesX, testingExamplesY)


def generateExamples(graph, testProportion=0.1, seed=0, filename=None):
    """
    Generates testing and training examples for node pairs from the given graph. Outputs
    results to text file at 'filename' if filename is not None. Each line is formatted as

    feature_1 ... feature_n classification

    with a newline separating the training examples from the testing examples

    Args:
        graph (snap.TNGraph or snap.TUNGraph) : the graph to generate examples from
        testingProportion (float) : the proportion of edges to use in the testing set
        seed (hashable) : a seed for the random number generator
        filename (string) : file location to write results to

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

    print "Generating edge features"
    neighborTable = {}
    for node in graph.Nodes():
        neighbors = set([ nodeID for nodeID in node.GetOutEdges() ]) | set([ nodeID for nodeID in node.GetInEdges()])
        neighborTable[node.GetId()] = len(neighbors)

    examplesX = [ featureextraction.extractFeatures(graph, edge.GetSrcNId(), edge.GetDstNId(), neighborTable) for edge in graph.Edges() ]
    examplesY = np.ones(int(graph.GetEdges()))

    random.shuffle(examplesX)
    trainingExamplesX = examplesX[:splitIndex]
    testingExamplesX = examplesX[splitIndex:]

    # Next, generate pairs of nodes which are not edges to balance out the training and testing sets
    # For the purpose of balancing against existing edges, we randomly choose from among pairs of node
    # which share a neighbor but do not form an edge
    print "Generating non-edge samples"
    nodeIDs = [ node.GetId() for node in graph.Nodes() ]
    nodePairs = []

    for srcID in nodeIDs:
        srcNode = graph.GetNI(srcID)
        srcOutNeighbors = frozenset([ nodeID for nodeID in srcNode.GetOutEdges() ])
        srcInNeighbors = frozenset([ nodeID for nodeID in srcNode.GetInEdges() ])
        srcNeighbors = srcOutNeighbors | srcInNeighbors

        secondDegrees = set()
        for neighbor in srcNeighbors:
            neighborNode = graph.GetNI(neighbor)
            nOutNeighbors = frozenset([ nodeID for nodeID in neighborNode.GetOutEdges() ])
            nInNeighbors = frozenset([ nodeID for nodeID in neighborNode.GetInEdges() ])
            nNeighbors = nOutNeighbors | nInNeighbors
            secondDegrees = secondDegrees | nNeighbors
        secondDegrees = secondDegrees - set([srcID])

        for dstID in secondDegrees:
            if not graph.IsEdge(srcID, dstID):
                nodePairs.append((srcID, dstID))

    nodePairs = random.sample(nodePairs, graph.GetEdges())

    """
    while len(nodePairs) < graph.GetEdges():
        pair = tuple(random.sample(nodeIDs, 2))
        if not graph.IsEdge(pair[0], pair[1]):
            nodePairs.add(pair)
    """

    print "Generating non-edge features"
    nonEdgeExamplesX = [ featureextraction.extractFeatures(graph, pair[0], pair[1], neighborTable) for pair in nodePairs ]
    nonEdgeExamplesY = np.zeros(int(graph.GetEdges()))

    trainingExamplesX += nonEdgeExamplesX[:splitIndex]
    testingExamplesX  += nonEdgeExamplesX[splitIndex:]
    trainingExamplesY =  np.concatenate((examplesY[:splitIndex],nonEdgeExamplesY[:splitIndex]), axis = 0)
    testingExamplesY  =  np.concatenate((examplesY[splitIndex:],nonEdgeExamplesY[splitIndex:]), axis = 0)

    #Instead of empty space between training and testing sets, write -1's.
    #This is used as a flag to easily find separation between data sets
    #while using np.genfromtext()
    (length,numFeatures) = np.shape(trainingExamplesX)
    separator = np.zeros(numFeatures) - 1

    # Write the examples to a file if necessary
    #Note that the features are written to a .features file, and the
    #labels are written to a .labels file
    if filename is not None:
        try:
            f = open(filename + '.features', 'w')
            for example in trainingExamplesX:
                f.write(" ".join([ str(val) for val in example]))
                f.write("\n")
            f.write(" ".join([ str(val) for val in separator]))
            f.write("\n")
            for example in testingExamplesX:
                f.write(" ".join([ str(val) for val in example ]))
                f.write("\n")
            f.close()

            f = open(filename + '.labels', 'w')
            for example in trainingExamplesY:
                f.write(str(example))
                f.write("\n")
            f.write("-1\n")
            for example in testingExamplesY:
                f.write(str(example))
                f.write("\n")
            f.close()

        except IOError as e:
            print "Error writing to file {0}: {1}".format(filename, e.strerror)
        
    return (trainingExamplesX,trainingExamplesY, testingExamplesX, testingExamplesY)


def generateFeatureFiles(filenames=None):
    """
    Generates the text files for given data sets.
    
    Args:
        filenames:  List of filenames which must names in allFileNames and have an associated load method
                    If no argument is given, generate the feature files for all data sets
    """
    if filenames is None: 
        filenames = allFileNames
    for fn in filenames:
        if fn in allFileNames:
            print "Generating feature files for: " + fn
            sys.stdout.flush()
            exec('graph = load' + fn + '()')
            generateExamples(graph,filename = 'featureFiles/' + fn)
            sys.stdout.flush()
    
        else:
            print "Error generating feature files for: " + fn
