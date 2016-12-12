import snap
import sys
import math

NUM_FEATURES = 20
SRC_D_IN, \
SRC_D_OUT, \
SRC_IN_OUT, \
SRC_OUT_IN, \
DST_D_IN, \
DST_D_OUT, \
DST_IN_OUT, \
DST_OUT_IN, \
NEIGHBORS, \
TRIAD_0, \
TRIAD_1, \
TRIAD_2, \
TRIAD_3, \
TRIAD_0_N, \
TRIAD_1_N, \
TRIAD_2_N, \
TRIAD_3_N, \
JACCARD, \
ADAMIC, \
PREFERENTIAL = range(NUM_FEATURES)

def extractFeatures(graph, srcNodeID, dstNodeID):
    """
    Extracts a set of features for an edge from srcNodeID to dstNodeID
    in the graph.

    Features:
        SRC_D_IN : in-degree of src
        SRC_D_OUT : out-degree of src
        SRC_IN_OUT : in-degree / out-degree of src
        SRC_OUT_IN : out-degree / in-degree of src
        DST_D_IN : in-degree of dst
        DST_D_OUT : out-degree of dst
        DST_IN_OUT : in-degree / out-degree of dst
        DST_OUT_IN : out-degree / in-degree of dst
        NEIGHBORS : number of nodes adjacent to both src and dst
        TRIAD_0 : participation in triads of form src -> neighbor -> dst
        TRIAD_1 : participation in triads of form src -> neighbor <- dst
        TRIAD_2 : participation in triads of form src <- neighbor -> dst
        TRIAD_3 : participation in triads of form src <- neighbor <- dst
        TRIAD_0_N : TRIAD_0 / NEIGHBORS ratio
        TRIAD_1_N : TRIAD_1 / NEIGHBORS ratio
        TRIAD_2_N : TRIAD_2 / NEIGHBORS ratio
        TRIAD_3_N : TRIAD_3 / NEIGHBORS ratio
        JACCARD : Jaccard's coefficient (common neighbors / all neighbors)
        ADAMIC : Adamic/Adar coefficient
        PREFERENTIAL : preferential attachment coefficient (neighbors(src) * neighbors(dst))

    Args:
        graph (snap.TNGraph or snap.TUNGraph): the graph to extract features from
        srcNodeID (int): the ID of the source of the edge
        dstNodeID (int): the ID for the destination of the edge

    Returns:
        A list of features extracted from the graph for the specified edge
    """

    features = [0] * NUM_FEATURES

    srcNode = graph.GetNI(srcNodeID)
    dstNode = graph.GetNI(dstNodeID)

    # Basic feature extraction
    features[SRC_D_IN] = srcNode.GetInDeg()
    features[SRC_D_OUT] = srcNode.GetOutDeg()
    features[SRC_IN_OUT] = features[SRC_D_IN]  * 1.0 / (1.0 + features[SRC_D_OUT]) #Add 1 to denominator so we don't divide by 0
    features[SRC_OUT_IN] = features[SRC_D_OUT] * 1.0 / (1.0 + features[SRC_D_IN])
    features[DST_D_IN] = dstNode.GetInDeg()
    features[DST_D_OUT] = dstNode.GetOutDeg()
    features[DST_IN_OUT] = features[DST_D_IN]  * 1.0 / (1.0 + features[DST_D_OUT])
    features[DST_OUT_IN] = features[DST_D_OUT] * 1.0 / (1.0 + features[DST_D_IN])

    # Path lengths. Set to sys.maxint if no path exists
    #srcDstPath = snap.GetShortPath(graph, srcNodeID, dstNodeID, True)
    #dstSrcPath = snap.GetShortPath(graph, dstNodeID, srcNodeID, True)

    #features[SRC_DST_PATH] = srcDstPath if srcDstPath != -1 else sys.maxsize
    #features[DST_SRC_PATH] = dstSrcPath if dstSrcPath != -1 else sys.maxsize

    # Calculate the number of shared neighbors
    srcOutNeighbors = frozenset([ nodeID for nodeID in srcNode.GetOutEdges() ])
    srcInNeighbors = frozenset([ nodeID for nodeID in srcNode.GetInEdges() ])
    srcNeighbors = srcOutNeighbors | srcInNeighbors

    dstOutNeighbors = frozenset([ nodeID for nodeID in dstNode.GetOutEdges() ])
    dstInNeighbors = frozenset([ nodeID for nodeID in dstNode.GetInEdges() ])
    dstNeighbors = dstOutNeighbors | dstInNeighbors

    neighbors = srcNeighbors & dstNeighbors
    features[NEIGHBORS] = len(neighbors)

    # Calculate the triad participation
    features[TRIAD_0] = len(srcOutNeighbors & dstInNeighbors)
    features[TRIAD_1] = len(srcOutNeighbors & dstOutNeighbors)
    features[TRIAD_2] = len(srcInNeighbors & dstInNeighbors)
    features[TRIAD_3] = len(srcInNeighbors & dstOutNeighbors)

    features[TRIAD_0_N] = features[TRIAD_0] * 1.0 / (1.0 + features[NEIGHBORS])
    features[TRIAD_1_N] = features[TRIAD_1] * 1.0 / (1.0 + features[NEIGHBORS])
    features[TRIAD_2_N] = features[TRIAD_2] * 1.0 / (1.0 + features[NEIGHBORS])
    features[TRIAD_3_N] = features[TRIAD_3] * 1.0 / (1.0 + features[NEIGHBORS])

    # Calculate graph distance coefficients
    features[JACCARD] = len(neighbors) / len(srcNeighbors | dstNeighbors)
    adamicCoeff = 0
    for neighborID in neighbors:
        neighbor = graph.GetNI(neighborID)
        nNeighbors = set([ nodeID for nodeID in neighbor.GetOutEdges() ]) | set([ nodeID for nodeID in neighbor.GetInEdges()])
        if len(nNeighbors) > 1:
            adamicCoeff += 1.0 / math.log(len(nNeighbors))
    features[ADAMIC] = adamicCoeff
    features[PREFERENTIAL] = len(srcNeighbors) * len(dstNeighbors)

    return features
