import os
import glob
import networkx as nx
import numpy as np

def clear_output_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    files = glob.glob(os.path.join(directory, "*"))
    for f in files:
        os.remove(f)

def initialize_graph(edges, thresholds):
    G = nx.DiGraph()
    G.add_nodes_from(thresholds.keys())

    for u, v, w in edges:
        G.add_edge(u, v, influence=w)

    for node, threshold in thresholds.items():
        G.nodes[node]['threshold'] = threshold

    return G

def _initialize_thresholds(G):
    for n in G.nodes():
        if 'threshold' not in G._node[n]:
            G._node[n]['threshold'] = np.random.rand()
        elif G._node[n]['threshold'] > 1:
            raise Exception("Node threshold cannot be larger than 1")

def _initialize_influences(G):
    in_deg = G.in_degree()
    for e in G.edges():
        if 'influence' not in G[e[0]][e[1]]:
            G[e[0]][e[1]]['influence'] = 1.0 / in_deg[e[1]]
        elif G[e[0]][e[1]]['influence'] > 1:
            raise Exception("Edge influence cannot be larger than 1")
