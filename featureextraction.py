import snap
import sys

SRC_D_IN = 0
SRC_D_OUT = 1
DST_D_IN = 2
DST_D_OUT = 3
NEIGHBORS = 4
TRIAD_0 = 5
TRIAD_1 = 6
TRIAD_2 = 7
TRIAD_3 = 8
NUM_FEATURES = 9

def extractFeatures(graph, srcNodeID, dstNodeID):
    """
    Extracts a set of features for an edge from srcNodeID to dstNodeID
    in the graph.

    Features:
        SRC_D_IN : in-degree of src
        SRC_D_OUT : out-degree of src
        DST_D_IN : in-degree of dst
        DST_D_OUT : out-degree of dst
        NEIGHBORS : number of nodes adjacent to both src and dst
        TRIAD_0 : participation in triads of form src -> neighbor -> dst
        TRIAD_1 : participation in triads of form src -> neighbor <- dst
        TRIAD_2 : participation in triads of form src <- neighbor -> dst
        TRIAD_3 : participation in triads of form src <- neighbor <- dst

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
    features[DST_D_IN] = dstNode.GetInDeg()
    features[DST_D_OUT] = dstNode.GetOutDeg()

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

    return features
