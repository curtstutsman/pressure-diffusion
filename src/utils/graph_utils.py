import networkx as nx
import numpy as np
import random
import gzip
import io
import requests
import pickle
import os
from src.utils.path_utils import ROOT


def create_erdos_renyi_graph(num_nodes, edge_probability):
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

    print(f"Number of nodes: {graph.number_of_nodes()}")
    print(f"Number of edges: {graph.number_of_edges()}")

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

def generate_stochastic_block_model(num_nodes):
    sizes = [num_nodes // 3, num_nodes // 3, num_nodes - 2 * (num_nodes // 3)]
    probs = [
        [0.0125, 0.00125, 0.00125],  # Block 1
        [0.00125, 0.0125, 0.00125],   # Block 2
        [0.00125, 0.00125, 0.0125]   # Block 3
    ] 
    
    # Generate the stochastic block model graph
    G = nx.stochastic_block_model(sizes, probs)
    
    # Convert to a directed graph
    return G.to_directed()


def generate_parent_dominant_graph(num_nodes, edge_prob):
    edge_prob = edge_prob
    graph = nx.DiGraph()
    for node in range(num_nodes):
        num_parents = int(num_nodes * edge_prob / 2)
        parents = np.random.choice(range(num_nodes), size=num_parents, replace=False)
        for parent in parents:
            graph.add_edge(parent, node)
    return graph

def sample_louvain_facebook():
    # URL of the Facebook network dataset
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    pickle_file = "./data/raw_networks/facebook.pkl"

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
    partition = community_louvain.best_partition(G)
    communities = list(set(partition.values()))
    sampled_nodes = [node for node, comm in partition.items() if comm == communities[0]]
    return G.subgraph(sampled_nodes).to_directed()

def create_facebook_graph(nodes=0):
    # URL of the Facebook network dataset
    url = "https://snap.stanford.edu/data/facebook_combined.txt.gz"
    pickle_file = ROOT + "/data/raw_networks/facebook.pkl"

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

    if nodes:
        random_nodes = random.sample(list(G.nodes()), nodes)
        G = G.subgraph(random_nodes).copy()

    # Now you can use the graph without re-downloading
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    G = G.to_directed()
    return G

def create_wiki_graph(nodes=0):
    # URL of the Wikipedia network dataset
    url = "https://snap.stanford.edu/data/wiki-Vote.txt.gz"
    pickle_file = ROOT + "/data/raw_networks/wiki.pkl"

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

    if nodes:
        random_nodes = random.sample(list(G.nodes()), nodes)
        G = G.subgraph(random_nodes).copy()

    # Now you can use the graph without re-downloading
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    G = G.to_directed()
    return G


def create_bitcoin_graph(nodes=0):
    # URL of the Facebook network dataset
    url = "https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz"
    pickle_file = ROOT + "/data/raw_networks/bitcoin.pkl"

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
            G = nx.DiGraph(name="Bitcoin‑OTC")
            for line in f:
                src, dst, rating, ts = line.strip().split(',')
                G.add_edge(int(src), int(dst))
            
        # Save the graph to a pickle file for future use
        with open(pickle_file, 'wb') as f:
            pickle.dump(G, f)
        print("Graph downloaded and saved to pickle file.")

    if nodes:
        random_nodes = random.sample(list(G.nodes()), nodes)
        G = G.subgraph(random_nodes).copy()

    # Now you can use the graph without re-downloading
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    G = G.to_directed()
    return G

def generate_watts_strogatz(n=5000, k=8, p=0.01):
    """
    Generate a Watts-Strogatz small-world graph.
    n: number of nodes
    k: each node is connected to k nearest neighbors in a ring topology
    p: probability of rewiring each edge
    """
    G = nx.watts_strogatz_graph(n, k, p)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    G = G.to_directed()
    return G


def generate_planted_partition(l=50, k=100, p_in=0.05, p_out=0.0001):
    """
    Generate a planted partition graph (stochastic block model).
    l: number of communities
    k: number of nodes per community
    p_in: probability of edges inside communities
    p_out: probability of edges between communities (set small for weak expansion)
    """
    G = nx.planted_partition_graph(l, k, p_in, p_out)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    G = G.to_directed()
    return G


def generate_barabasi_albert(n=5000, m=5):
    """
    Generate a Barabási–Albert scale-free graph.
    n: number of nodes
    m: number of edges to attach from a new node to existing nodes
    """
    G = nx.barabasi_albert_graph(n, m)
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    G = G.to_directed()
    return G
