import os
import matplotlib.pyplot as plt
import networkx as nx

def visualize_diffusion(G, active_nodes, output_dir, budget):
    plt.figure(figsize=(18, 7))
    pos = nx.spring_layout(G, seed=0)
    labels = nx.get_edge_attributes(G, 'influence')
    node_colors = ['red' if node in active_nodes else 'blue' for node in G.nodes()]

    nx.draw(G, pos, node_color=node_colors, node_size=500)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f'{v:.2f}' for k, v in labels.items()}, font_size=7)
    node_numbers = {n: str(n) for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_numbers, font_size=10, font_color='white')
    # node_thresholds = {n: f'{G._node[n]["threshold"]:.2f}' for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, font_size=5, verticalalignment='bottom')

    output_path = os.path.join(output_dir, f"graph_budget_{budget}.png")
    plt.savefig(output_path)
    plt.close()
