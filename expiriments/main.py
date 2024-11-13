import argparse
import numpy as np
import networkx as nx
import requests
import gzip
import os
import pickle
import io
from diffusion_models.pressure_diffusion import pressure_linear_threshold
from utils.graph_utils import clear_output_directory
from scripts.heuristic import compute_heuristic
from scripts.greedy import greedy_im
from utils.visualization import visualize_diffusion
from scripts.influence import influence

def parse_arguments():
    """Parse command-line arguments for the diffusion model."""
    parser = argparse.ArgumentParser(description="Run the Linear Threshold Diffusion Model.")
    
    parser.add_argument(
        "--alpha", type=float, default=0,
        help="Set the alpha parameter for influence adjustment (default: 0.1)"
    )
    parser.add_argument(
        "--budget", type=int, default=1,
        help="Set the budget for size of seed set"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Enable or disable visualization (default: True)"
    )
    parser.add_argument(
        "--model", type=str, default="pressure_threshold",
        help="Diffusion model to use"
    )
    parser.add_argument(
        "--output_dir", type=str, default="",
        help="directory to output graph to"
    )
    
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()
    if args.output_dir:
        clear_output_directory(args.output_dir)

    # URL of the Facebook network dataset
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    pickle_file = "facebook.pkl"

    # Check if the graph is already saved as a pickle file
    if os.path.exists(pickle_file):
        # Load the graph from the pickle file
        with open(pickle_file, 'rb') as f:
            G = pickle.load(f)
        print("Graph loaded from pickle file.")
    else:
        # Download and read the data from the URL
        response = requests.get(url)
        with gzip.open(io.BytesIO(response.content), 'rt') as f:
            G = nx.read_edgelist(f, create_using=nx.Graph(), nodetype=int)
        
        # Save the graph to a pickle file for future use
        with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)
        print("Graph downloaded and saved to pickle file.")

    # Now you can use the graph without re-downloading
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    G = G.to_directed()

    # init thresholds
    for n in G.nodes():
        if 'threshold' not in G._node[n]:
            G._node[n]['threshold'] = np.random.rand(1)[0]   
        
    #Run heuristic
    heuristic_set = compute_heuristic(G, args.budget, args.alpha)
    heuristic_influence = influence(G, heuristic_set, args.model, args.alpha)
    print("Pressure Threshold heuristic values:")
    print(heuristic_set, heuristic_influence)

    heuristic_set = compute_heuristic(G, args.budget, 0)
    heuristic_influence = influence(G, heuristic_set, args.model, 0)
    print("Base Threshold heuristic values:")
    print(heuristic_set, heuristic_influence)

    #Run diffusion
    best_seed_set, max_influence = greedy_im(G, budget=args.budget, diffusion_model=args.model, alpha=args.alpha)
    print("Greedy results for pressure threshold")
    print(best_seed_set, max_influence )

    best_seed_set, max_influence = greedy_im(G, budget=args.budget, diffusion_model="linear_threshold", alpha=args.alpha)
    print("Greedy results for base threshold")
    print(best_seed_set, max_influence )

    # if args.output_dir:
    #     visualize_diffusion(G, sum(active_nodes, []), args.output_dir, args.budget)
