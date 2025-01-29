import numpy as np
import copy
import networkx as nx

def pressure_linear_threshold_optimized(G, seeds, alpha=0, steps=0):
    """
    Optimized version of the pressure linear threshold model.

    Produces the same final set of activated nodes (and thus the same total
    influence/spread) as the original 'pressure_linear_threshold' function
    given the same inputs, but with reduced Python overhead.

    Parameters
    ----------
    G : networkx Graph/DiGraph
        The graph over which diffusion occurs.
        If not directed, a directed copy is created internally.
    seeds : list of nodes
        The seed nodes (initially active).
    alpha : float, default=0
        The parameter for adjusting outgoing influence once a node is activated.
    steps : int, default=0
        Number of rounds to simulate; if <= 0, runs until no new activations.

    Returns
    -------
    layers : list of lists
        layers[0]: the seed nodes (initially active).
        layers[i]: the nodes activated at round i.

    Notes
    -----
    - Each node should have a 'threshold' attribute in [0,1]. If not present,
      it is assigned randomly in [0,1].
    - Each directed edge (u->v) should have an 'influence' attribute in [0,1].
      If not, it defaults to 1/in_degree(v).
    """

    # 1) Validate seeds
    for s in seeds:
        if s not in G:
            raise ValueError(f"Seed node {s} is not in the graph.")

    # 2) Convert to directed if needed, and copy so we do not mutate the original
    if not G.is_directed():
        DG = G.to_directed()
    else:
        DG = copy.deepcopy(G)

    # 3) Initialize thresholds if missing; check validity
    for n in DG.nodes():
        if 'threshold' not in DG.nodes[n]:
            DG.nodes[n]['threshold'] = np.random.rand()

    # ----------------------------------------------------------------------
    # 5) Prepare for diffusion: We'll replicate the same iterative approach
    #    as the original: "round by round," calling an equivalent of
    #    _diffuse_one_round each time until no change (or steps exhausted).
    # ----------------------------------------------------------------------
    A_list = list(seeds)  # list of currently active nodes (in order)
    A_set  = set(seeds)   # set for O(1) membership checks
    layers = [ A_list[:] ]  # layers[0]: seeds

    # A helper to do exactly the same "activate -> immediate influence update" logic
    def _diffuse_one_round(DG, active_list, active_set, alpha_val):
        newly_activated_set = set()

        # For each active node in the current list
        for src in active_list:
            # Check all successors
            for nb in DG.successors(src):
                if nb in active_set:
                    continue  # already active
                # sum influences from active predecessors
                infl_sum = 0.0
                for pred in DG.predecessors(nb):
                    if pred in active_set:
                        infl_sum += DG[pred][nb]['influence']

                # check threshold
                if infl_sum >= DG.nodes[nb]['threshold']:
                    # Mark this neighbor as newly activated
                    newly_activated_set.add(nb)
                    # Immediately adjust outgoing edges from 'nb'
                    if alpha_val > 0:
                        old_infl_sum = infl_sum
                        for out_nb in DG.successors(nb):
                            if out_nb not in active_set:
                                old_val = DG[nb][out_nb]['influence']
                                new_val = old_val + alpha_val * old_infl_sum
                                DG[nb][out_nb]['influence'] = min(1.0, new_val)

        if newly_activated_set:
            # Extend the active list, update the set
            newly_activated_list = list(newly_activated_set)
            active_list.extend(newly_activated_list)
            active_set.update(newly_activated_set)
        else:
            newly_activated_list = []

        return newly_activated_list

    # ----------------------------------------------------------------------
    # 6) Diffuse for up to 'steps' rounds, or until no new activations
    # ----------------------------------------------------------------------
    round_count = 0
    while True:
        if steps > 0 and round_count >= steps:
            # We have done the user-requested # of rounds
            break

        # Perform one diffusion round
        newly_activated = _diffuse_one_round(DG, A_list, A_set, alpha)

        # Keep track of which nodes were newly activated this round
        layers.append(newly_activated)

        if not newly_activated:
            # no more nodes got activated => done
            break

        round_count += 1
        if len(A_set) == len(DG):
            # all nodes active => done
            break

    return layers
