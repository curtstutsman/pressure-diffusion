"""
Experiment: Evaluating Seed Selection by Greedy Algorithm Between Pressure Threshold and Linear Threshold

Author: Curt Stutsman
Date: 12/2/2024

Description:
    The purpose of this script is to compare the performance of the greedy algorithm on the Pressure Threshold (PT) and Linear Threshold (LT)
    diffusion models across four different graph types. This script will randomize edge weights and thresholds between simulations, 
    and for each pair of graph type and diffusion model, it will output the top 10 most frequently selected seed nodes and the average influence.

    Graphs Evaluated:
    1. Random
    2. Louvaine Community Sample from SNAP'S Facebook
    3. Stochastic Block Model
    4. Parent-Dominant 

Goals:
- Simulate diffusion for multiple runs per graph type for each diffusion model.
- Randomize edge weights and node thresholds for each simulation.
- Save average influence spread and top nodes per graph and model to a CSV file.

Inputs:
--nodes:        Number of nodes in the graph.
--edge_prob:    Probability of an edge occurring between any two nodes.
--alpha:        Pressure parameter alpha to control influence propagation.
--k:            Number of seed nodes to select.
--simulations:  Number of simulations to run per heuristic.

Outputs:
- Results are written to a CSV file after simulations.
"""

#=======================================
#   Library Imports
#=======================================
import pandas as pd
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from src.scripts.greedy import greedy_im
from src.scripts.influence import influence
from src.utils.graph_utils import (
    create_directed_graph, 
    add_random_thresholds,
    generate_parent_dominant_graph,
    sample_louvain_facebook,
    generate_stochastic_block_model,
)
from src.scripts.weighted_network import weighted_network

#=======================================
#   Internal Function Defines
#=======================================
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate Greedy Algorithm on Pressure and Linear Threshold Models.")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes in the graph.")
    parser.add_argument("--edge_prob", type=float, required=True, help="Edge probability in the graph.")
    parser.add_argument("--alpha", type=float, required=True, help="Pressure parameter for influence propagation.")
    parser.add_argument("--k", type=int, required=True, help="Number of seed nodes to select.")
    parser.add_argument("--simulations", type=int, required=True, help="Number of simulations to run.")
    return parser.parse_args()

def simulate(graph, model_type, k, alpha):
    """
    Perform a single simulation with randomized weights and thresholds using the greedy algorithm.
    """
    random_graph = add_random_thresholds(graph)
    seed_set, spread = greedy_im(random_graph, k,  model_type, alpha=alpha)
    return seed_set, spread

def create_graph(graph_type, num_nodes, edge_prob):
    """
    Create a graph based on the specified type.
    """
    if graph_type == "random":
        return create_directed_graph(num_nodes, edge_prob)
    elif graph_type == "parent_dominant":
        return generate_parent_dominant_graph(num_nodes, 2*edge_prob)
    elif graph_type == "louvain_facebook":
        return sample_louvain_facebook()
    elif graph_type == "stochastic_block":
        return generate_stochastic_block_model(num_nodes)
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")

#=======================================
#   Main Execution
#=======================================
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    num_nodes = args.nodes
    edge_prob = args.edge_prob
    alpha = args.alpha
    budget = args.k
    num_simulations = args.simulations
    output_dir = "data/results"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, "seed_selection_by_mdoel.csv")

    # Delete existing results file
    if os.path.exists(results_file):
        os.remove(results_file)

    # Models and graph types to evaluate
    models = ["linear_threshold", "pressure_threshold"]
    graph_types = ["random"]

    # Create a fresh results file
    pd.DataFrame(columns=["Model", "Graph Type", "Top 10 Seed Nodes", "Counts", "Average Influence"]).to_csv(results_file, index=False)

    # Run experiments
    for graph_type in graph_types:
        base_graph = create_graph(graph_type, num_nodes, edge_prob)
        base_graph = weighted_network(base_graph, 'wc')
        for model in models:
            print(f"Running {num_simulations} simulations for {model} on {graph_type} graph...")
            
            # Store seed counts and influence results
            all_seeds = []
            total_influence = 0

            # Use ProcessPoolExecutor for parallel simulation
            with ProcessPoolExecutor() as executor:
                simulation_results = list(executor.map(
                    simulate,
                    [base_graph] * num_simulations,
                    [model] * num_simulations,
                    [budget] * num_simulations,
                    [alpha] * num_simulations,
                ))

            # Process simulation results
            for seed_set, spread in simulation_results:
                all_seeds.extend(seed_set)
                total_influence += spread

            # Analyze top 10 seed nodes
            seed_counts = Counter(all_seeds).most_common(10)
            avg_influence = total_influence / num_simulations

            # Format results
            top_seeds = [seed for seed, _ in seed_counts]
            counts = [count for _, count in seed_counts]
            new_result = {
                "Model": model,
                "Graph Type": graph_type,
                "Top 10 Seed Nodes": top_seeds,
                "Counts": counts,
                "Average Influence": avg_influence
            }

            # Save to CSV
            results_df = pd.read_csv(results_file)
            results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
            results_df.to_csv(results_file, index=False)

            print(f"Completed {model} on {graph_type} graph. Average Influence: {avg_influence}")
            print(f"Top 10 Seed Nodes: {top_seeds} with Counts: {counts}")