def degree(graph, k, alpha=0):
    """
    Selects the top-k nodes based on degree, optionally adjusted by alpha for pressure.

    Parameters:
        graph (nx.Graph or nx.DiGraph): The input graph.
        k (int): Number of nodes to select.
        alpha (float): Pressure parameter to amplify incoming influence.

    Returns:
        list: List of top-k nodes selected by degree.
    """
    degrees = {
        node: graph.out_degree(node) + (alpha) * graph.in_degree(node)
        for node in graph.nodes()
    }

    # Select top-k nodes by adjusted degree
    return sorted(degrees, key=degrees.get, reverse=True)[:k]


def degree_discount(graph, k, alpha=0):
    """
    Selects the top-k nodes using degree discount heuristic, optionally adjusted by alpha for pressure.

    Parameters:
        graph (nx.Graph or nx.DiGraph): The input graph.
        k (int): Number of nodes to select.
        alpha (float): Pressure parameter to amplify incoming influence.

    Returns:
        list: List of top-k nodes selected by degree discount heuristic.
    """
    degrees = {
        node: graph.out_degree(node) + (alpha) * graph.in_degree(node)
        for node in graph.nodes()
    }
    t = {node: 0 for node in graph.nodes()}  # Tracks selected neighbors
    discounted_degrees = degrees.copy()
    selected_nodes = []

    for _ in range(k):
        # Select the node with the highest discounted degree
        u = max(discounted_degrees, key=discounted_degrees.get)
        selected_nodes.append(u)

        # Update discounted degrees for neighbors of u
        for neighbor in graph.neighbors(u):
            if neighbor not in selected_nodes:
                t[neighbor] += 1
                discounted_degrees[neighbor] = degrees[neighbor] - t[neighbor]

        # Remove selected node from consideration
        discounted_degrees.pop(u)

    return selected_nodes