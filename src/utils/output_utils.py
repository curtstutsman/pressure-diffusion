import os
import pickle
import pandas as pd

def save_results_to_csv(results, output_dir, filename):
    """
    Save results to a CSV file.

    Parameters:
        results (dict): Experiment results.
        output_dir (str): Directory to save the file.
        filename (str): Name of the file.
    """
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, filename), index=False)

def save_graph_to_pickle(graph, pickle_dir, filename):
    """
    Save a graph to a pickle file.

    Parameters:
        graph (nx.Graph): The graph object.
        pickle_dir (str): Directory to save the file.
        filename (str): Name of the file.
    """
    os.makedirs(pickle_dir, exist_ok=True)
    with open(os.path.join(pickle_dir, filename), "wb") as f:
        pickle.dump(graph, f)