import copy
import argparse
import networkx as nx
import numpy as np
from utils.visualization import visualize_diffusion
from utils.graph_utils import _initialize_influences, _initialize_thresholds
from utils.graph_utils import initialize_graph, clear_output_directory

def pressure_linear_threshold(G, seeds, steps=0, alpha=0, output_dir=""):
    """
    Simulate the pressure threshold diffusion model.
    """

    if type(G) == nx.MultiGraph or type(G) == nx.MultiDiGraph:
      raise Exception( \
          "linear_threshold() is not defined for graphs with multiedges.")
    
    for s in seeds:
        if s not in G.nodes():
            raise Exception("seed", s, "is not in graph")
        
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
        elif DG[e[0]][e[1]]['influence'] > 1:
            raise Exception("edge influence:", DG[e[0]][e[1]]['influence'], \
            "cannot be larger than 1")

    A = copy.deepcopy(seeds)
    if output_dir:
        visualize_diffusion(DG, A, 0, output_dir)  # Initial state

    if steps <= 0:
        return _diffuse_all(DG, A, alpha, output_dir)
    return _diffuse_k_rounds(DG, A, steps, alpha, output_dir)


def _diffuse_all(G, A, alpha, output_dir):

    layer_i_nodes = [list(A)]
    step = 1

    while True:
        len_old = len(A)
        A, activated_nodes = _diffuse_one_round(G, A, alpha)
        layer_i_nodes.append(activated_nodes)

        if output_dir:
            visualize_diffusion(G, A, step, output_dir)
            step += 1

        if len(A) == len_old:
            break

    return layer_i_nodes


def _diffuse_k_rounds(G, A, steps, alpha, output_dir):

    layer_i_nodes = [list(A)]
    step = 1

    while steps > 0 and len(A) < len(G):
        len_old = len(A)
        A, activated_nodes = _diffuse_one_round(G, A, alpha)
        layer_i_nodes.append(activated_nodes)

        if output_dir:
            visualize_diffusion(G, A, step, output_dir)
            step += 1

        if len(A) == len_old:
            break
        steps -= 1

    return layer_i_nodes


def _diffuse_one_round(G, A, alpha):
    activated_nodes = set()

    # for active_node in A:
    #     total_influence = _influence_sum(G, list(set(G.predecessors(active_node)).intersection(A)), active_node)
    #     _adjust_outgoing_influence(G, active_node, alpha, total_influence, A)

    for s in A:
        for nb in G.successors(s):
            if nb in A or nb in activated_nodes:
                continue
            active_nb = list(set(G.predecessors(nb)).intersection(A))
            total_influence = _influence_sum(G, active_nb, nb)
            if total_influence >= G._node[nb]['threshold']:
                activated_nodes.add(nb)
                _adjust_outgoing_influence(G, nb, alpha, total_influence, A)

    A.extend(list(activated_nodes))
    return A, list(activated_nodes)

def _influence_sum(G, froms, to):
    return sum(G[f][to]['influence'] for f in froms)

def _adjust_outgoing_influence(G, node, alpha, total_influence, A):
    for neighbor in G.successors(node):
        if neighbor not in A:
            G[node][neighbor]['influence'] = min(
                1.0, G[node][neighbor]['influence'] + alpha * total_influence
            )

