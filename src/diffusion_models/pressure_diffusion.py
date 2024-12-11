import copy
import argparse
import networkx as nx
import numpy as np
from src.utils.visualization import visualize_diffusion

def pressure_linear_threshold(G, seeds, steps=0, alpha=0, output_dir=""):
    """
    Simulate the pressure threshold diffusion model.
    """

    if G.is_multigraph():
        raise ValueError("pressure_linear_threshold() is not defined for multigraphs.")

    # Validate seeds
    missing_seeds = [s for s in seeds if s not in G]
    if missing_seeds:
        raise ValueError(f"Seeds {missing_seeds} are not present in the graph.")

    # Convert to directed graph if not already
    DG = G.to_directed() if not G.is_directed() else G.copy()

    # Initialize thresholds
    thresholds = {}
    for n in DG.nodes():
        threshold = DG.nodes[n].get('threshold', np.random.rand())
        if threshold > 1:
            raise ValueError(f"Node {n} has threshold {threshold} > 1.")
        thresholds[n] = threshold
    nx.set_node_attributes(DG, thresholds, 'threshold')

    # Initialize activated set as a set for O(1) lookups
    activated = set(seeds)
    layer_i_nodes = [list(activated)]

    if steps <= 0:
        return _diffuse_all(DG, activated, alpha, layer_i_nodes)
    return _diffuse_k_rounds(DG, activated, steps, alpha, layer_i_nodes)


def _diffuse_all(G, A, alpha, layer_i_nodes):
  while True:
    len_old = len(A)
    A, activated_nodes_of_this_round = _diffuse_one_round(G, A, alpha)
    layer_i_nodes.append(activated_nodes_of_this_round)
    if len(A) == len_old:
      break
  return layer_i_nodes


def _diffuse_k_rounds(G, A, steps, alpha, layer_i_nodes):
  while steps > 0 and len(A) < len(G):
    len_old = len(A)
    A, activated_nodes_of_this_round = _diffuse_one_round(G, A, alpha)
    layer_i_nodes.append(activated_nodes_of_this_round)
    if len(A) == len_old:
      break
    steps -= 1
  return layer_i_nodes


def _diffuse_one_round(G, A, alpha):
  activated_nodes_of_this_round = set()
  for s in A:
    nbs = G.successors(s)
    for nb in nbs:
      if nb in A:
        continue
      active_nb = list(set(G.predecessors(nb)).intersection(set(A)))
      incoming_influence = _influence_sum(G, active_nb, nb)
      if incoming_influence >= G._node[nb]['threshold']:
        activated_nodes_of_this_round.add(nb)
        _adjust_outgoing_influence(G, nb, alpha, incoming_influence, A)

  A.update(activated_nodes_of_this_round)
  return A, list(activated_nodes_of_this_round)

def _influence_sum(G, froms, to):
    influence_sum = 0.0
    for f in froms:
        influence_sum += G[f][to]['influence']
    return influence_sum

def _adjust_outgoing_influence(G, node, alpha, total_influence, A):
    for neighbor in G.successors(node):
        if neighbor not in A:
            G[node][neighbor]['influence'] = min(
                1.0, G[node][neighbor]['influence'] + alpha * total_influence
            )

