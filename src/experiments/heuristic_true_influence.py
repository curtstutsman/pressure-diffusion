"""
Experiment: Evaluating True Influence Performance of Heuristics on Influence Maximization

Author: Curt Stutsman
Date: 11/18/2024

Description:
    This script tests the performance of various seed selection heuristics on two diffusion models:
    1. The Linear Threshold (LT) Model
    2. The Pressure Threshold (PT) Model

    Heuristics Evaluated (all described in /data/results/heuristic.txt):
    1. Degree Heuristic (α=0)
    2. Degree Heuristic (𝛼≠0)
    3. Degree Discount Heuristic (α=0)
    4. Degree Discount Heuristic (𝛼≠0)
    5. Two-Step Pressure Heuristic
    6. Pressure Flow Heuristic
    7. Amplified Coverage Heuristic
    8. Pressure Degree Discount Heuristic
    9. Adaptive Combined Heuristic
    10. Local Pressure Heuristic
    11. Hybrid Degree Heuristic
    12. Adjusted Degree Discount Heuristic

    Graphs Evaluated:
    1. Random
    2. Louvaine Community Sample from SNAP'S Facebook
    3. Stochastic Block Model
    4. Parent-Dominant 

Goals:
- Simulate diffusion for multiple runs per heuristic for each model.
- Randomize edge weights and node thresholds for each simulation.
- Save average influence spread per heuristic and model to a CSV file.

Inputs:
--nodes:        Number of nodes in the graph.
--edge_prob:    Probability of an edge occurring between any two nodes.
--alpha:        Pressure parameter alpha to control influence propagation.
--k:            Number of seed nodes to select.
--simulations:  Number of simulations to run per heuristic.

Outputs:
- Results are written to a CSV file after each heuristic finishes all its simulations.
"""

#=======================================
#   Library Imports
#=======================================
import pandas as pd
import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from src.scripts.heuristic import (
    DegreeHeuristic,
    DegreeDiscountHeuristic,
    TwoStepPressureHeuristic,
    PressureFlowHeuristic,
    AmplifiedCoverageHeuristic,
    PressureDegreeDiscountHeuristic,
    AdaptiveCombinedHeuristic, 
    LocalPressureHeuristic,
    HybridDegreeHeuristic,
    AdjustedDegreeDiscountHeuristic
)
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
    parser = argparse.ArgumentParser(description="Evaluate Heuristics on Influence Maximization.")
    parser.add_argument("--nodes", type=int, required=True, help="Number of nodes in the graph.")
    parser.add_argument("--edge_prob", type=float, required=True, help="Edge probability in the graph.")
    parser.add_argument("--alpha", type=float, required=True, help="Pressure parameter for influence propagation.")
    parser.add_argument("--k", type=int, required=True, help="Number of seed nodes to select.")
    parser.add_argument("--simulations", type=int, required=True, help="Number of simulations to run.")
    return parser.parse_args()

def simulate(graph, heuristic, budget, model_type, alpha):
    """
    Perform a single simulation with randomized weights and thresholds.
    """
    randomized_graph = add_random_thresholds(graph.copy())
    seed_set = heuristic.select(randomized_graph, budget)
    return influence(randomized_graph, seed_set, model_type, alpha=alpha)

def create_graph(graph_type, num_nodes, edge_prob):
    """
    Create a graph based on the specified type.

    Parameters:
        graph_type (str): The type of graph to generate.
        num_nodes (int): Number of nodes in the graph.
        edge_prob (float): Probability of edge creation for random graphs.

    Returns:
        nx.DiGraph: A directed graph.
    """
    if graph_type == "random":
        return create_directed_graph(num_nodes, edge_prob)
    elif graph_type == "parent_dominant":
        return generate_parent_dominant_graph(num_nodes, 2 * edge_prob)
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
    results_file = os.path.join(output_dir, "heuristic_true_influence.csv")

    # Delete existing results file
    if os.path.exists(results_file):
        os.remove(results_file)

    # Create directed graph (static structure)
    base_graph = create_directed_graph(num_nodes, edge_prob)

    # Define heuristics
    heuristics = {
        "Degree": DegreeHeuristic(alpha=0),
        "Degree Discount": DegreeDiscountHeuristic(alpha=0),
        "Degree  w/ alpha": DegreeHeuristic(alpha=1),
        "Degree Discount w/ alpha": DegreeDiscountHeuristic(alpha=1),
        "Two Step": TwoStepPressureHeuristic(),
        "Pressure Flow": PressureFlowHeuristic(),                           
        "Amplified Coverage": AmplifiedCoverageHeuristic(alpha=alpha),
        "Pressure Degree Discount": PressureDegreeDiscountHeuristic(alpha=1),
        "Adaptive Combined" : AdaptiveCombinedHeuristic(alpha=alpha, weights=(1,1,0.5)),
        "Local Pressure" : LocalPressureHeuristic(alpha=alpha),
        "Local Pressure alpha=1" : LocalPressureHeuristic(alpha=1),
        "Hybdrid Degree" : HybridDegreeHeuristic(alpha=alpha),
        "Adjusted Degree Discount" : AdjustedDegreeDiscountHeuristic(alpha=alpha),          
    }

    # Models to evaluate
    models = ["pressure_threshold"]

    # Graph Types
    graph_types = ["parent_dominant", "random", "louvain_facebook", "stochastic_block"]

    # Create a fresh results file
    pd.DataFrame(columns=["Model", "Graph Type", "Heuristic", "Average Influence"]).to_csv(results_file, index=False)

    # Run experiments
    for model in models:
        for graph_type in graph_types:
            base_graph = create_graph(graph_type, num_nodes, edge_prob)
            for name, heuristic in heuristics.items():
                print(f"Running {num_simulations} simulations for {name} on {graph_type} graph...")
                # Use ProcessPoolExecutor for parallel simulation
                with ProcessPoolExecutor() as executor:
                    simulation_results = list(executor.map(
                        simulate,
                        [base_graph] * num_simulations,
                        [heuristic] * num_simulations,
                        [budget] * num_simulations,
                        [model] * num_simulations,
                        [alpha] * num_simulations,
                    ))

                # Calculate average influence spread
                avg_spread = sum(simulation_results) / num_simulations
                new_result = {"Model": model, "Graph Type": graph_type, "Heuristic": name, "Average Influence": avg_spread}

                # Update results file
                results_df = pd.read_csv(results_file)
                results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
                results_df.to_csv(results_file, index=False)

                print(f"Completed {name} on {graph_type} graph. Average Influence: {avg_spread}")
