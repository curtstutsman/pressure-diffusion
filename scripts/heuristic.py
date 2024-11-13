import networkx as nx
import copy
import numpy as np

def compute_heuristic(G, k, alpha):

    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # init thresholds
    for n in DG.nodes():
        if 'threshold' not in DG._node[n]:
            DG._node[n]['threshold'] = np.random.rand(1)[0]    
        elif DG._node[n]['threshold'] > 1:
            raise Exception("node threshold:", DG._node[n]['threshold'], \
                "cannot be larger than 1")
        
    # init influences
    for e in DG.edges():
        if 'influence' not in DG[e[0]][e[1]]:
            DG[e[0]][e[1]]['influence'] = np.random.rand(1)[0]

    S = set()              # Seed set S
    dd = {}                # Degree discount for each node

    # Initial degree setup
    for v in G.nodes():
        dd[v] = G.degree(v) + alpha * G.in_degree(v)  # Initial degree discount is just the degree of the node

    # Greedily select k nodes
    for _ in range(k):
        # Select the node u with the highest degree discount that is not in S
        u = max((v for v in G.nodes() if v not in S), key=lambda v: dd[v])
        S.add(u)

        # Update the degree discount for the neighbors of u
        for v in G.neighbors(u):
            if v not in S:
                # Decrement the degree discount of each neighbor by 1
                dd[v] -= 1

    return list(S)