import typing as t
from cynetdiff.models import DiffusionModel
from tqdm import tqdm, trange
import heapq


def celf(
    model: DiffusionModel, n: int, k: int, num_trials: int = 1_000
) -> t.Tuple[t.Set[int], t.List[float]]:
    """
    Input: graph object, number of seed nodes
    Output: optimal seed set, resulting spread, time for each iteration
    Code adapted from this blog post:
    https://hautahi.com/im_greedycelf
    """

    # Run the CELF algorithm
    marg_gain = []

    # First, compute all marginal gains
    for node in trange(n):
        marg_gain.append(
            (
                -model.compute_marginal_gains([node], [], num_trials)[0],
                node,
            )
        )

    # Convert to heap
    heapq.heapify(marg_gain)

    max_mg, selected_node = heapq.heappop(marg_gain)
    seeds = [selected_node]
    spread = -max_mg
    spreads = [spread]

    # Greedily select remaining nodes
    for _ in trange(k - 1):
        while True:
            current_mg, current_node = heapq.heappop(marg_gain)
            new_mg_neg = -model.compute_marginal_gains(seeds, [current_node], num_trials)[1]

            if new_mg_neg <= current_mg:
                break
            else:
                heapq.heappush(marg_gain, (new_mg_neg, current_node))

        spread += -new_mg_neg
        seeds.append(current_node)
        spreads.append(spread)
        print (spread)
        if (spread >= 4039.0):
            break

    # Return the maximizing set S and the increasing spread values.
    return seeds, spreads