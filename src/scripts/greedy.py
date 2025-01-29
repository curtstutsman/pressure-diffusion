from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold
from src.diffusion_models.pressure_diffusion import pressure_linear_threshold
from src.diffusion_models.test  import pressure_linear_threshold_optimized
import networkx as nx
import numpy as np

def greedy_im(network, budget, diffusion_model, alpha=0):
    """
    Greedy Algorithm for selecting the best seed set of a user-specified budget.
    Assumes simulation looping and averaging are handled by the experiment script.

    Parameters:
        - network: A NetworkX graph object.
        - budget: Number of seed nodes to select.
        - diffusion_model: The diffusion model ("independent_cascade", "linear_threshold", or "pressure_threshold").
        - alpha: Pressure diffusion model parameter (optional, default is 0).

    Returns:
        - best_seed_set: List of selected seed nodes.
        - total_influence: Total influence spread for the selected seed set.
    """
    nodes = list(nx.nodes(network))
    max_influence = []
    best_seed_set = [] 

    for _ in range(budget):
        # Nodes not yet selected
        nodes_to_try = list(set(nodes) - set(best_seed_set))
        influence = np.zeros(len(nodes_to_try))

        for i in range(len(nodes_to_try)):
            # Add the candidate node to the seed set
            best_seed_set_plus_ith_node = \
                        list(set(best_seed_set + [nodes_to_try[i]]))

            # Simulate the diffusion process
            if diffusion_model == "independent_cascade":
                layers = independent_cascade(network, best_seed_set_plus_ith_node)
            elif diffusion_model == "linear_threshold":
                layers = linear_threshold(network, best_seed_set_plus_ith_node)
            elif diffusion_model == "pressure_threshold":
                layers = pressure_linear_threshold_optimized(network, best_seed_set_plus_ith_node, alpha=alpha)
            else:
                raise ValueError(f"Unknown diffusion model: {diffusion_model}")

            for k in range(len(layers)):
                influence[i] = influence[i] + len(layers[k])

        # Select the node with the highest marginal influence
        max_influence.append(np.max(influence))    
        best_seed_set.append(nodes_to_try[np.argmax(influence)])

    print(best_seed_set)
    print(max(max_influence))

    return best_seed_set, max(max_influence)
