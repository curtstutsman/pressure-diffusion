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
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.scripts.greedy import greedy_im
from src.utils.graph_utils import create_facebook_graph
from src.scripts.weighted_network import weighted_network
from src.scripts.heuristic import (
    DegreeHeuristic,
    AmplifiedCoverageHeuristic
)
from src.scripts.influence import influence
import multiprocessing

# Global Constants
MODEL = 'pressure_threshold'
OUTPUT_DIR = "data/results"
MAX_BUDGET = 1

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
        "--nodes", type=int, required=False, default=0, help="Number of nodes to include in subgraph of facebook graph."
    )
    parser.add_argument(
        "--simulations", type=int, required=True, help="Number of simulations to run per heuristic."
    )
    parser.add_argument(
        "--output", type=str, default="main.csv", help="Output CSV file name."
    )
    return parser.parse_args()

#=======================================
#   Simulation Function
#=======================================
def simulate(simulation_task):
    """
    Perform a single simulation with the given algorithm and budget.

    Args:
        simulation_task (dict): Contains network, algorithm, budget, alpha.

    Returns:
        float: The influence spread result.
    """
    network, algorithm, budget, alpha = simulation_task['network'], simulation_task['algorithm'], simulation_task['budget'], simulation_task['alpha']
    
    if algorithm == 'greedy':
        _, spread = greedy_im(network, budget, MODEL, alpha)
    else:
        if algorithm == 'degree':
            heuristic = DegreeHeuristic(alpha=0)
        elif algorithm == 'amplified_coverage':
            heuristic = AmplifiedCoverageHeuristic(alpha=alpha)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
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
    nodes = args.nodes
    output_file = args.output

    # Generate facebook networkx graph with edged weights = 1/in_degree
    network_unweighted = create_facebook_graph(nodes)
    network = weighted_network(network_unweighted, 'wc')

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, output_file)

     # Initialize or create results file with headers if it doesn't exist
    if not os.path.exists(results_file):
        pd.DataFrame(columns=["Model", "Algorithm", "Budget", "Average Influence"]).to_csv(results_file, index=False)

    # Algorithms / Heuristics to test
    algorithms = ['degree', 'amplified_coverage', 'greedy']

    # Determine number of workers (using all available CPUs)
    num_workers = multiprocessing.cpu_count()
    print(f"Utilizing {num_workers} CPUs")

     # Initialize ProcessPoolExecutor once
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Iterate over budgets and algorithms
        for k in range(1, MAX_BUDGET + 1):
            for algorithm in algorithms:
                print(f"Running {num_simulations} simulations for {algorithm} with budget = {k}...")
                
                # Prepare the arguments for each simulation
                simulation_tasks = [{
                    'network': network,
                    'algorithm': algorithm,
                    'budget': k,
                    'alpha': alpha
                } for _ in range(num_simulations)]
                
                total_influence = 0.0

                # Submit all tasks and collect futures
                futures = [executor.submit(simulate, task) for task in simulation_tasks]
                
                # As each future completes, accumulate the results
                for future in as_completed(futures):
                    try:
                        spread = future.result()
                        total_influence += spread
                    except Exception as exc:
                        print(f"Simulation generated an exception: {exc}")
                        # Depending on requirements, you might want to handle exceptions differently

                # Calculate average influence
                avg_influence = total_influence / num_simulations

                # Prepare the result entry
                new_result = {
                    "Model": MODEL,
                    "Algorithm": algorithm,
                    "Budget": k,
                    "Average Influence": avg_influence
                }

                # Append the new result to the CSV file
                results_df = pd.DataFrame([new_result])
                results_df.to_csv(results_file, mode='a', header=False, index=False)

                print(f"Completed {algorithm} with budget = {k}. Average Influence: {avg_influence}")

    print(f"All simulations completed. Results saved to {results_file}")