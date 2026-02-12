import typing as t
import heapq
from tqdm import trange
import cynetdiff
from cynetdiff.models import DiffusionModel


class Candidate:
    """
    Helper class for CELF++ that stores a candidate node, its current marginal gain,
    a flag indicating the size of the seed set when its gain was computed,
    and a 'best' field (the node responsible for its last evaluation).
    """
    def __init__(self, node: int, mg: float, flag: int, best: int):
        self.node = node
        self.mg = mg
        self.flag = flag
        self.best = best

    def __repr__(self):
        return f"Candidate(node={self.node}, mg={self.mg}, flag={self.flag}, best={self.best})"


def celf_pp(
    model: DiffusionModel,
    n: int,
    k: int,
    num_trials: int = 1_000
) -> t.Tuple[t.List[int], t.List[float]]:
    """
    CELF++ algorithm for influence maximization using the cynetdiff API.

    Parameters:
        model (DiffusionModel): An instance of a diffusion model. It must implement:
            - compute_marginal_gains(seed_set: List[int], candidate_nodes: List[int], num_trials: int)
              When candidate_nodes is empty, it returns a list (e.g. [gain_value]);
              when candidate_nodes is non-empty, it returns a tuple where the marginal gain is at index [1].
        n (int): Total number of nodes in the network (assumed nodes are labeled 0 to n-1).
        k (int): The desired number of seed nodes.
        num_trials (int): The number of simulation trials used in computing marginal gains.

    Returns:
        A tuple (seeds, spreads) where:
            - seeds is a list of selected seed node IDs.
            - spreads is a list of cumulative influence spread values after each seed is added.
    """

    # Helper function to extract the marginal gain from the API's return value.
    def get_gain(seed_set: t.List[int], candidate_nodes: t.List[int]) -> float:
        result = model.compute_marginal_gains(seed_set, candidate_nodes, num_trials)
        if not candidate_nodes:
            # When candidate_nodes is empty, assume the API returns a list with one element.
            return result[0] if isinstance(result, list) and len(result) > 0 else result
        else:
            # When candidate_nodes is non-empty, assume the API returns a tuple with the gain at index 1.
            return result[1] if isinstance(result, (list, tuple)) and len(result) > 1 else result

    heap = []  # We'll use a max-heap by pushing negative marginal gains with a tie-breaker.

    # --- Initialization: Compute initial marginal gains for every node.
    for node in trange(n, desc="Initializing candidates"):
        mg = get_gain([node], [])
        cand = Candidate(node=node, mg=mg, flag=0, best=node)
        # Use a tuple with (negative marginal gain, candidate.node as tie-breaker, candidate)
        heapq.heappush(heap, (-mg, cand.node, cand))

    # --- Select the first seed.
    neg_mg, _, top_candidate = heapq.heappop(heap)
    seeds = [top_candidate.node]
    total_spread = -neg_mg  # Because mg was stored as a negative value in the heap.
    spreads = [total_spread]

    # --- Greedy selection for the remaining seeds.
    for _ in trange(k - 1, desc="Selecting seeds"):
        while True:
            current_neg_mg, _, current_candidate = heapq.heappop(heap)
            # Check if the candidate's stored marginal gain is valid:
            # either it was computed with the current seed set (flag equals len(seeds))
            # or its cached "best" node is already in the seed set.
            if current_candidate.flag == len(seeds) or current_candidate.best in seeds:
                break
            else:
                # Its value is stale; recompute its marginal gain given the current seed set.
                new_mg = get_gain(seeds, [current_candidate.node])
                current_candidate.mg = new_mg
                current_candidate.flag = len(seeds)
                current_candidate.best = current_candidate.node  # reset best to itself
                heapq.heappush(heap, (-new_mg, current_candidate.node, current_candidate))
        # Accept the candidate whose marginal gain is up-to-date.
        seeds.append(current_candidate.node)
        total_spread += current_candidate.mg
        spreads.append(total_spread)

    return seeds, spreads
