from src.diffusion_models.independent_cascade import independent_cascade
from src.diffusion_models.linear_threshold import linear_threshold
from src.diffusion_models.pressure_diffusion import pressure_linear_threshold
from src.scripts.weighted_network import weighted_network
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
    best_seed_set = []
    total_influence = 0

    for _ in range(budget):
        # Nodes not yet selected
        nodes_to_try = list(set(nodes) - set(best_seed_set))
        influence = np.zeros(len(nodes_to_try))

        for i, node in enumerate(nodes_to_try):
            # Add the candidate node to the seed set
            seed_set = best_seed_set + [node]

            # Simulate the diffusion process
            if diffusion_model == "independent_cascade":
                layers = independent_cascade(network, seed_set)
            elif diffusion_model == "linear_threshold":
                layers = linear_threshold(network, seed_set)
            elif diffusion_model == "pressure_threshold":
                layers = pressure_linear_threshold(network, seed_set, alpha=alpha)
            else:
                raise ValueError(f"Unknown diffusion model: {diffusion_model}")

            # Compute the total spread in this single simulation
            influence[i] = sum(len(layer) for layer in layers)

        # Select the node with the highest marginal influence
        best_node = nodes_to_try[np.argmax(influence)]
        best_seed_set.append(best_node)

        # Update the total influence based on the node selected
        total_influence += np.max(influence)

    return best_seed_set, total_influence
