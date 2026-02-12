#!/usr/bin/env python3
"""
LT vs PT diffusion demo (paper-friendly) with per-step frames saved separately.

What it does:
- Builds a random directed graph (default n=10) with a NEW random seed each run
  (unless you pass --seed for reproducibility)
- Node thresholds ~ Uniform(0,1)
- Edge weights (influence) = 1 / in_degree(target)  (classic LT normalization)
- Runs diffusion to completion for:
    * LT: alpha=0
    * PT: alpha>0
- Saves frames per time step into:
    out/LT/frame_00.png, frame_01.png, ...
    out/PT/frame_00.png, frame_01.png, ...

Paper readability changes:
- Larger nodes so "id + θ" fits
- More spaced layout (spring_layout with larger k and more iterations)
- bbox_inches="tight" for clean crops

Optional:
- If Graphviz layout is available (pygraphviz + graphviz installed), will use sfdp
  to reduce overlaps even more; otherwise falls back to spring_layout.

Usage:
  python demo.py
  python demo.py --alpha 0.35
  python demo.py --seed 12345
  python demo.py --p 0.20 --n 10
"""

import os
import time
import copy
import argparse
import shutil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# ----------------------------
# Diffusion (LT / PT)
# ----------------------------

def pressure_linear_threshold(G, seeds, alpha=0.0, steps=0):
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise Exception("Not defined for graphs with multiedges.")
    for s in seeds:
        if s not in G:
            raise Exception(f"Seed {s} not in graph")

    DG = copy.deepcopy(G) if G.is_directed() else G.to_directed()

    # Thresholds should exist; fallback kept for safety
    for n in DG.nodes():
        if "threshold" not in DG.nodes[n]:
            DG.nodes[n]["threshold"] = float(np.random.rand())

    # Influences should exist; fallback kept for safety
    in_deg = dict(DG.in_degree())
    for u, v in DG.edges():
        if "influence" not in DG[u][v]:
            deg = in_deg.get(v, 0)
            DG[u][v]["influence"] = 0.0 if deg == 0 else 1.0 / deg

    A = list(seeds)

    if steps <= 0:
        layers, graph_history = _diffuse_all_with_history(DG, A, alpha)
    else:
        layers, graph_history = _diffuse_k_rounds_with_history(DG, A, alpha, steps)

    return DG, layers, graph_history


def _diffuse_all_with_history(G, A, alpha):
    layers = [list(A)]
    graph_history = [copy.deepcopy(G)]  # snapshot at step 0 (before any PT boosts)

    while True:
        before = len(A)
        A, newly = _diffuse_one_round(G, A, alpha)
        layers.append(newly)
        graph_history.append(copy.deepcopy(G))  # snapshot after this round's boosts

        if len(A) == before:
            break

    return layers, graph_history


def _diffuse_k_rounds_with_history(G, A, alpha, steps):
    layers = [list(A)]
    graph_history = [copy.deepcopy(G)]

    while steps > 0 and len(A) < len(G):
        before = len(A)
        A, newly = _diffuse_one_round(G, A, alpha)
        layers.append(newly)
        graph_history.append(copy.deepcopy(G))

        if len(A) == before:
            break
        steps -= 1

    return layers, graph_history



def _diffuse_one_round(G, A, alpha):
    active = set(A)
    newly = set()

    for s in list(active):
        for nb in G.successors(s):
            if nb in active:
                continue
            active_preds = set(G.predecessors(nb)).intersection(active)
            total = sum(G[p][nb]["influence"] for p in active_preds)
            if total >= G.nodes[nb]["threshold"]:
                newly.add(nb)
                _adjust_outgoing_influence(G, nb, alpha, total, active)

    A.extend(sorted(newly))
    return A, sorted(newly)


def _adjust_outgoing_influence(G, node, alpha, total_influence, active_set):
    # PT: reinforce only edges to nodes not yet active (matches your described PT)
    if alpha <= 0:
        return
    for nb in G.successors(node):
        if nb in active_set:
            continue
        G[node][nb]["influence"] = min(
            1.0, G[node][nb]["influence"] + alpha * total_influence
        )


# ----------------------------
# Graph construction
# ----------------------------

def build_random_demo_graph(n, p, rng):
    # Use a fresh internal seed so each run produces a different structure
    g_seed = int(rng.integers(0, 2**31 - 1))
    G = nx.gnp_random_graph(n, p, seed=g_seed, directed=True)
    G.remove_edges_from(nx.selfloop_edges(G))
    return G


def assign_uniform_thresholds(G, rng):
    for v in G.nodes():
        G.nodes[v]["threshold"] = float(rng.uniform(0.0, 1.0))


def assign_influence_one_over_indegree(G):
    # Ensure every existing edge has influence = 1/in_degree(target)
    in_deg = dict(G.in_degree())
    for u, v in G.edges():
        deg = in_deg[v]
        G[u][v]["influence"] = 0.0 if deg == 0 else 1.0 / deg


# ----------------------------
# Layout: reduce overlaps
# ----------------------------

def compute_layout(G, seed):
    """
    Try Graphviz (sfdp) for better separation if available.
    Fallback to spring_layout with larger spacing.
    # """
    # try:
    #     from networkx.drawing.nx_agraph import graphviz_layout
    #     return graphviz_layout(G, prog="sfdp")
    # except Exception:
    #     # Larger k => more spacing. More iterations => more stable.
    return nx.spring_layout(G, k=3.0, iterations=450)

# ----------------------------
# Visualization
# ----------------------------

def active_after_t(layers, t):
    t = min(t, len(layers) - 1)
    out = set()
    for i in range(t + 1):
        out.update(layers[i])
    return out

def draw_edge_labels_tight(ax, pos, edge_labels, label_pos=0.3, fontsize=10):
    """
    Draw edge labels tightly bound to edges (manual placement).

    pos: node -> (x, y)
    edge_labels: dict {(u, v): label}
    """
    for (u, v), text in edge_labels.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        # linear interpolation along edge
        x = x1 * (1 - label_pos) + x2 * label_pos
        y = y1 * (1 - label_pos) + y2 * label_pos

        ax.text(
            x, y, text,
            fontsize=fontsize,
            color="black",
            ha="center",
            va="center",
            bbox=dict(
                boxstyle="round,pad=0.15",
                fc="white",
                ec="none",
                alpha=0.85,
            ),
            zorder=5,        # stay above edges
            clip_on=True,    # don’t float outside axes
        )


def visualize_frame(
    DG,
    pos,
    active_nodes,
    newly_activated,
    seeds,          # unused, kept for signature consistency
    step_idx,
    out_path,
    highlight_boosted_edges=False,
):
    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    ax.axis("off")

    active = set(active_nodes)
    newly = set(newly_activated)

    inactive = set(DG.nodes()) - active
    active_old = active - newly  # active but not newly activated

    # --- sizing (paper scale) ---
    NODE_SIZE = 3100
    NODE_SIZE_NEW = 3300
    LABEL_FONT_SIZE = 13
    EDGE_LABEL_FONTSZ = 12

    # --- edge style (normal black arrows) ---
    BASE_EDGE_WIDTH = 2.6
    BOOST_EDGE_WIDTH = 5.0
    ARROW_SIZE_BASE = 26
    ARROW_SIZE_BOOST = 34

    # Draw all edges (black, normal thickness)
    nx.draw_networkx_edges(
        DG,
        pos,
        ax=ax,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=ARROW_SIZE_BASE,
        width=BASE_EDGE_WIDTH,
        alpha=0.85,
        edge_color="black",
        # connectionstyle="arc3,rad=0.15",
        min_source_margin=22,
        min_target_margin=26,
    )

    seed_set = set(seeds)
    # Draw boosted edges on top (PT only) as thicker black
    if highlight_boosted_edges and newly:
        active_before = set(active_nodes) - set(newly_activated)  # active at start of this round
        boosted_edges = [(u, v) for u in newly if u not in seed_set for v in DG.successors(u) if v not in active_before]

        if boosted_edges:
            # --- bottom pass: red outline ---
            nx.draw_networkx_edges(
                DG,
                pos,
                ax=ax,
                edgelist=boosted_edges,
                arrows=True,
                arrowstyle="-|>",
                arrowsize=28,
                width=2.8,                 # thick outline
                edge_color="#C62828",      # deep red
                # connectionstyle="arc3,rad=0.15",
                alpha=1.0,
                min_source_margin=22,
                min_target_margin=26,
            )

    # --- edge labels (weights) ---
    e_labels = nx.get_edge_attributes(DG, "influence")
    nx.draw_networkx_edge_labels(
        DG,
        pos,
        ax=ax,
        edge_labels={k: f"{v:.2f}" for k, v in e_labels.items()},
        font_size=EDGE_LABEL_FONTSZ,
        rotate=False,
        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.80),
        label_pos=0.3,
    )

    # --- nodes ---
    nx.draw_networkx_nodes(
        DG, pos, ax=ax,
        nodelist=sorted(inactive),
        node_size=NODE_SIZE,
        node_color="#4C78A8",
        edgecolors="white",
        linewidths=2.0,
    )

    nx.draw_networkx_nodes(
        DG, pos, ax=ax,
        nodelist=sorted(active_old),
        node_size=NODE_SIZE,
        node_color="#E45756",
        edgecolors="white",
        linewidths=2.0,
    )

    # newly activated: same fill (red) but thick black border
    if newly:
        nx.draw_networkx_nodes(
            DG, pos, ax=ax,
            nodelist=sorted(newly),
            node_size=NODE_SIZE_NEW,
            node_color="#E45756",
            edgecolors="black",
            linewidths=5.0,
        )

    # --- node labels: id + threshold ---
    labels = {n: f"{n}\nθ={DG.nodes[n]['threshold']:.2f}" for n in DG.nodes()}
    nx.draw_networkx_labels(
        DG, pos, ax=ax,
        labels=labels,
        font_color="white",
        font_size=LABEL_FONT_SIZE,
    )

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)




def render_sequence(graph_history, layers, pos, seeds, out_dir, is_pt=False):
    os.makedirs(out_dir, exist_ok=True)

    # graph_history[t] corresponds to the graph state after step t updates
    # layers[t] corresponds to newly active at step t
    T = min(len(graph_history), len(layers))

    for t in range(T):
        DG_t = graph_history[t]
        active = active_after_t(layers, t)
        newly = layers[t]
        out_path = os.path.join(out_dir, f"frame_{t:02d}.png")

        visualize_frame(
            DG=DG_t,
            pos=pos,
            active_nodes=active,
            newly_activated=newly,
            seeds=seeds,
            step_idx=t,
            out_path=out_path,
            highlight_boosted_edges=is_pt,
        )



def enforce_min_distance(pos, min_dist=1.2, iters=200):
    nodes = list(pos.keys())
    for _ in range(iters):
        moved = False
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                a, b = nodes[i], nodes[j]
                d = pos[a] - pos[b]
                dist = float(np.linalg.norm(d)) + 1e-9
                if dist < min_dist:
                    push = (min_dist - dist) * (d / dist) * 0.5
                    pos[a] = pos[a] + push
                    pos[b] = pos[b] - push
                    moved = True
        if not moved:
            break
    return pos



# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--p", type=float, default=0.15)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--out", type=str, default="demo_out")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional fixed RNG seed for reproducibility.")
    parser.add_argument("--seeds", type=str, default="0,1",
                        help="Comma-separated seed nodes, e.g. '0,1'")
    args = parser.parse_args()

    # --- clean output directory on each run ---
    if os.path.exists(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    # New graph each run by default
    run_seed = args.seed if args.seed is not None else int(time.time_ns() % (2**31 - 1))
    rng = np.random.default_rng(run_seed)

    # Build base graph + shared attributes (fair comparison within the run)
    G = build_random_demo_graph(args.n, args.p, rng)
    assign_uniform_thresholds(G, rng)
    assign_influence_one_over_indegree(G)

    # Layout fixed per run (so frames don’t jump)
    layout_seed = int(rng.integers(0, 2**31 - 1))
    pos = compute_layout(G, layout_seed)
    pos = enforce_min_distance(pos, min_dist=1.4, iters=200)


    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    for s in seeds:
        if s < 0 or s >= args.n:
            raise ValueError(f"Seed {s} out of range for n={args.n}")

    # LT and PT start from the same initial graph
    G_lt = copy.deepcopy(G)
    G_pt = copy.deepcopy(G)

    DG_lt, layers_lt, hist_lt = pressure_linear_threshold(G_lt, seeds=seeds, alpha=0.0, steps=0)
    DG_pt, layers_pt, hist_pt = pressure_linear_threshold(G_pt, seeds=seeds, alpha=args.alpha, steps=0)


    out_lt = os.path.join(args.out, "LT")
    out_pt = os.path.join(args.out, "PT")

    render_sequence(hist_lt, layers_lt, pos, seeds, out_dir=out_lt, is_pt=False)
    render_sequence(hist_pt, layers_pt, pos, seeds, out_dir=out_pt, is_pt=True)


    active_lt = set().union(*layers_lt)
    active_pt = set().union(*layers_pt)

    print("=== Run info ===")
    print(f"run_seed = {run_seed}")
    print(f"n={args.n} p={args.p} seeds={seeds} alpha={args.alpha}")
    print(f"LT frames: {len(layers_lt)} saved in {os.path.abspath(out_lt)}")
    print(f"PT frames: {len(layers_pt)} saved in {os.path.abspath(out_pt)}")
    print("=== Final spread ===")
    print(f"LT active: {len(active_lt)}/{DG_lt.number_of_nodes()}  PT active: {len(active_pt)}/{DG_pt.number_of_nodes()}")
    print(f"PT-only nodes: {sorted(active_pt - active_lt)}")
    print(f"LT-only nodes: {sorted(active_lt - active_pt)}")
    print("\nTip: if you like a run, rerun with --seed <run_seed> to reproduce it exactly.")

if __name__ == "__main__":
    main()
