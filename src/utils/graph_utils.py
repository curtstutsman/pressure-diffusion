import networkx as nx
import random

def create_directed_graph(num_nodes, edge_probability):
    """
    Creates a random directed graph with a specified number of nodes and edge probability.

    Parameters:
        num_nodes (int): Number of nodes in the graph.
        edge_probability (float): Probability of an edge between any two nodes.

    Returns:
        nx.DiGraph: A randomly generated directed graph.
    """
    if num_nodes <= 0:
        raise ValueError("Number of nodes must be greater than 0.")
    if not (0 <= edge_probability <= 1):
        raise ValueError("Edge probability must be between 0 and 1.")

    graph = nx.DiGraph()
    graph.add_nodes_from(range(num_nodes))

    for u in range(num_nodes):
        for v in range(num_nodes):
            if u != v:
                if random.random() < edge_probability:
                    graph.add_edge(u, v)

    return graph

def add_random_thresholds(graph):
    """
    Adds a random threshold between 0 and 1 to each node in the graph.

    Parameters:
        graph (nx.Graph or nx.DiGraph): The input graph.

    Returns:
        The modified graph
    """
    for node in graph.nodes():
        graph.nodes[node]['threshold'] = random.uniform(0, 1)

    return graph
