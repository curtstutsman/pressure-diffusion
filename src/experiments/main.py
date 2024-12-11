"""
Experiment: Evaluating True Influence Spread of Different IM-Solvers under Pressure Threshold

Author: Curt Stutsman
Date: 12/10/2024

Description:
    The purpose of this script is to compare the performance of the degree heuristic, the amplified coverage heuristic, and the greedy algorithm
    under the pressure threshold diffusion model. Each algorithm will have its influence evaluated on the facebook SNAPS sample graph (4,039 nodes, 88,234 edges)
    with budgets ranging (0, 20]

    Graphs Evaluated:
    1. SNAP's Facebook: https://snap.stanford.edu/data/ego-Facebook.html

Goals:
- Simulate diffusion for multiple IM solving algs/heuristics
- Compare performance as budget increases

Inputs:
--alpha:        Pressure parameter alpha to control influence propagation.
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
from src.scripts.greedy import greedy_im
from src.utils.graph_utils import create_facebook_graph
from src.scripts.weighted_network import weighted_network
from src.scripts.heuristic import (
    DegreeHeuristic,
    AmplifiedCoverageHeuristic
)
from src.scripts.influence import influence

# Global Constants
MODEL = 'pressure_threshold'
OUTPUT_DIR = "data/results"
MAX_BUDGET = 20

#=======================================
#   Argument Parsing
#=======================================
def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate Greedy Algorithm, Degree Heuristic, and Amplified Coverage Heuristic Under Pressure Diffusion."
    )
    parser.add_argument(
        "--alpha", type=float, required=True, help="Pressure parameter for influence propagation."
    )
    parser.add_argument(
        "--simulations", type=int, required=True, help="Number of simulations to run per heuristic."
    )
    return parser.parse_args()

#=======================================
#   Simulation Function
#=======================================
def simulate(args):
    """
    Perform a single simulation with the given algorithm and budget.

    Args:
        args (tuple): Contains (network, algorithm, budget, alpha)

    Returns:
        float: The influence spread result.
    """
    network, algorithm, budget, alpha = args
    if algorithm == 'greedy':
        # Greedy algorithm returns both seed_set and spread
        _, spread = greedy_im(network, budget, MODEL, alpha)
    else:
        # Instantiate the appropriate heuristic
        if algorithm == 'degree':
            heuristic = DegreeHeuristic(alpha=0)
        elif algorithm == 'amplified_coverage':
            heuristic = AmplifiedCoverageHeuristic(alpha=alpha)
        
        # Select seed set and calculate influence spread
        seed_set = heuristic.select(network, budget)
        spread = influence(network, seed_set, MODEL, alpha)
    
    return spread

#=======================================
#   Main Execution
#=======================================
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()
    alpha = args.alpha
    num_simulations = args.simulations

    # Generate facebook networkx graph with edged weights = 1/in_degree
    network = weighted_network(create_facebook_graph(), 'wc')

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, "main.csv")

    # Delete existing results file
    if os.path.exists(results_file):
        os.remove(results_file)

    # Algorithms / Heuristics to test
    algorithms = ['degree', 'amplified_coverage', 'greedy']

    # Create a fresh results file
    pd.DataFrame(columns=["Model", "Algorithm", "Budget", "Average Influence"]).to_csv(results_file, index=False)

    for k in range(1, MAX_BUDGET+1):
        for algorithm in algorithms:
            print(f"Running {num_simulations} simulations for {algorithm} with budget = {k}...")
            total_influence = 0

            # Prepare the arguments for each simulation
            simulation_args = [(network, algorithm, k, alpha) for _ in range(num_simulations)]
            
            # Execute simulations in parallel
            with ProcessPoolExecutor() as executor:
                # Using list to force evaluation and allow sum()
                simulation_results = list(executor.map(simulate, simulation_args))

            avg_influence = sum(simulation_results) / num_simulations

            new_result = {
                "Model": MODEL,
                "Algorithm": algorithm,
                "Budget" : k,
                "Average Influence": avg_influence
            }

            # Save to CSV
            results_df = pd.read_csv(results_file)
            results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
            results_df.to_csv(results_file, index=False)

            print(f"Completed {algorithm} with budget = {k}. Average Influence: {avg_influence}")