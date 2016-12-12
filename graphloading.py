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
    graph = snap.LoadEdgeList(snap.PNGraph, "datasets/wiki-Vote.txt", 0, 1, "\t")
    return pruneGraph(graph, 3)

def loadAstroPhysics():
    """
    Loads the Arxiv Astro Physics collaboration network

    Returns:
        A undirected graph (snap.TUNGraph)
    """
    graph = snap.LoadEdgeList(snap.PUNGraph, "datasets/ca-AstroPh.txt", 0, 1, "\t")
    return pruneGraph(graph, 3)

def loadCondensedMatter():
    """
    Loads the Arxiv Condensed Matter collaboration network

    Returns:
        A undirected graph (snap.TUNGraph)
    """
    graph = snap.LoadEdgeList(snap.PUNGraph, "datasets/ca-CondMat.txt", 0, 1, "\t")
    return pruneGraph(graph, 3)

def loadGeneralRelativity():
    """
    Loads the Arxiv General Relativity collaboration network

    Returns:
        A undirected graph (snap.TUNGraph)
    """
    graph = snap.LoadEdgeList(snap.PUNGraph, "datasets/ca-GrQc.txt", 0, 1, "\t")
    return pruneGraph(graph, 3)

def loadHEPhysics():
    """
    Loads the Arxiv High Energy Physics collaboration network

    Returns:
        A undirected graph (snap.TUNGraph)
    """
    graph = snap.LoadEdgeList(snap.PUNGraph, "datasets/ca-HepPh.txt", 0, 1, "\t")
    return pruneGraph(graph, 3)

def loadHEPhysicsTheory():
    """
    Loads the Arxiv High Energy Physics Theory  collaboration network

    Returns:
        A undirected graph (snap.TUNGraph)
    """
    graph = snap.LoadEdgeList(snap.PUNGraph, "datasets/ca-HepTh.txt", 0, 1, "\t")
    return pruneGraph(graph, 3)

def loadSlashdotOld():
    """
    Loads the Slashdot Zoo network from November 2008

    Returns:
        A directed graph (snap.TNGraph)
    """
    graph = snap.LoadEdgeList(snap.PNGraph, "datasets/soc-Slashdot0811.txt")
    return pruneGraph(graph, 5) 

def loadSlashdotNew():
    """
    Loads the Slashdot Zoo network from February 2009

    Returns:
        A directed graph (snap.TNGraph)
    """
    graph = snap.LoadEdgeList(snap.PNGraph, "datasets/soc-Slashdot0902.txt")
    return pruneGraph(graph, 5)

def pruneGraph(graph, minDegree):
    """
    Prunes a graph to remove nodes with degree less than minDegree

    Returns:
        A subgraph of graph
    """

    nIdV = snap.TIntV()
    for node in graph.Nodes():
        if node.GetDeg() > minDegree:
            nIdV.Add(node.GetId())
    return snap.GetSubGraph(graph, nIdV)

def generateExampleSplit(oldGraph, newGraph, testProportion=0.3, seed=None, filename=None):
    if seed is not None:
        random.seed(seed)

    # For the new graph, we only care about nodes which exist
    # in the old graph, so generate the subgraph first
    print "Pruning new graph nodes"
    oldNodeIDV = snap.TIntV()
    for node in oldGraph.Nodes():
        oldNodeIDV.Add(node.GetId())
    newGraph = snap.GetSubGraph(newGraph, oldNodeIDV)
    print "Old graph has {} nodes and {} edges".format(oldGraph.GetNodes(), oldGraph.GetEdges())
    print "New graph has {} nodes and {} edges".format(newGraph.GetNodes(), newGraph.GetEdges())

    print "Generating Adamic/Adar cache table"
    neighborTable = {}
    for node in oldGraph.Nodes():
        neighborTable[node.GetId()] = node.GetDeg()

    print "Generating edge features"
    # Generate edge features for edges which appear in the new graph but do not appear in the old graph
    edgeFeatures = [ featureextraction.extractFeatures(oldGraph, edge.GetSrcNId(), edge.GetDstNId(), neighborTable) for edge in newGraph.Edges()
                     if not oldGraph.IsEdge(edge.GetSrcNId(), edge.GetDstNId()) ]
    random.shuffle(edgeFeatures)
    print "Generated features for {} edges".format(len(edgeFeatures))

    print "Generating non-edge samples"
    # Generate pairs of nodes which share a neighbor but do not form an edge in the new graph or the old graph
    nodeIDs = [ node.GetId() for node in oldGraph.Nodes() ]
    nodePairs = []

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
            secondDegrees = secondDegrees | nOutNeighbors | nInNeighbors
        secondDegrees = secondDegrees - set([srcID])

        for dstID in secondDegrees:
            if not oldGraph.IsEdge(srcID, dstID) and not newGraph.IsEdge(srcID, dstID):
                nodePairs.append((srcID, dstID))
    print "Generated {} non-edge pairs".format(len(nodePairs))

    nodePairs = random.sample(nodePairs, min(9 * len(edgeFeatures), len(nodePairs)))
    print "Sampled {} non-edge pairs".format(len(nodePairs))
    
    print "Generating non-edge features"
    nonEdgeFeatures = [ featureextraction.extractFeatures(oldGraph, srcNode, dstNode, neighborTable) for srcNode, dstNode in nodePairs ]
    print "Generated features for {} non-edges".format(len(nonEdgeFeatures))

    edgeSplitIndex = int(len(edgeFeatures) * testProportion)
    nonEdgeSplitIndex = int(len(nonEdgeFeatures) * testProportion)

    testingEdgeFeatures = edgeFeatures[:edgeSplitIndex]
    testingNonEdgeFeatures = nonEdgeFeatures[:nonEdgeSplitIndex]
    trainingEdgeFeatures = edgeFeatures[edgeSplitIndex:]
    trainingNonEdgeFeatures = nonEdgeFeatures[nonEdgeSplitIndex:]

    testingExamplesX = testingEdgeFeatures + testingNonEdgeFeatures
    testingExamplesY = [1] * len(testingEdgeFeatures) + [0] * len(testingNonEdgeFeatures)
    trainingExamplesX = trainingEdgeFeatures + trainingNonEdgeFeatures
    trainingExamplesY = [1] * len(trainingEdgeFeatures) + [0] * len(trainingNonEdgeFeatures)
    
    print "Generate {} training examples and {} testing examples".format(len(trainingExamplesX), len(testingExamplesX))

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

def generateSlashdotExamples(testProportion=0.3, seed=None, filename=None):
  
    # Load in the graphs. For the new graph, we only care about nodes which exist
    # in the old graph, so generate the subgraph first
    oldGraph = loadSlashdotOld()
    newGraph = loadSlashdotNew()

    return generateExampleSplit(oldGraph, newGraph, testProportion, seed, filename)

def generateArtificialExamples(graph, newProportion=0.05, testProportion=0.3, seed=None, filename=None):

    # SNAP doesn't come with a graph copy method, so just induce an subgraph on the same set of nodes
    nodeIDVec = snap.TIntV()
    for node in graph.Nodes():
        nodeIDVec.Add(node.GetId())
    oldGraph = snap.GetSubGraph(graph, nodeIDVec)

    # Sample newProportion of edges and remove them the graph to artificially generate old graph
    sampleSize = int(graph.GetEdges() * newProportion)
    print "Sampling {} new edges out of {} total edges".format(sampleSize, graph.GetEdges())
    newEdges = random.sample([ (edge.GetSrcNId(), edge.GetDstNId()) for edge in graph.Edges() ], sampleSize)

    for srcNID, dstNID in newEdges:
        oldGraph.DelEdge(srcNID, dstNID)

    return generateExampleSplit(oldGraph, graph, testProportion, seed, filename)

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
        neighborTable[node.GetId()] = node.GetDeg()

    examplesX = [ featureextraction.extractFeatures(graph, edge.GetSrcNId(), edge.GetDstNId(), neighborTable) for edge in graph.Edges() ]
    examplesY = [1] * graph.GetEdges()

    random.shuffle(examplesX)
    trainingExamplesX = examplesX[:splitIndex]
    trainingExamplesY = examplesY[:splitIndex]
    testingExamplesX = examplesX[splitIndex:]
    testingExamplesY = examplesY[splitIndex:]

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

    print "Total node pairs: ", len(nodePairs)

    nodePairs = random.sample(nodePairs, min(10 * graph.GetEdges(), len(nodePairs)))

    """
    while len(nodePairs) < graph.GetEdges():
        pair = tuple(random.sample(nodeIDs, 2))
        if not graph.IsEdge(pair[0], pair[1]):
            nodePairs.add(pair)
    """

    print "Generating non-edge features"
    nonEdgeExamplesX = [ featureextraction.extractFeatures(graph, pair[0], pair[1], neighborTable) for pair in nodePairs ]
    nonEdgeExamplesY = [0] * len(nonEdgeExamplesX)
    splitIndex = len(nonEdgeExamplesX) - int(len(nonEdgeExamplesX) * testProportion)

    trainingExamplesX += nonEdgeExamplesX[:splitIndex]
    trainingExamplesY += nonEdgeExamplesY[:splitIndex]
    testingExamplesX  += nonEdgeExamplesX[splitIndex:]
    testingExamplesY  += nonEdgeExamplesY[splitIndex:]

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
            generateArtificialExamples(graph,filename = 'featureFiles/' + fn)
            sys.stdout.flush()
    
        else:
            print "Error generating feature files for: " + fn
