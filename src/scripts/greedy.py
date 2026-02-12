import typing as t
from cynetdiff.models import DiffusionModel
from tqdm import tqdm, trange
import heapq


def greedy_im(model: DiffusionModel, n: int, k: int, num_trials: int = 1_000):
    """
    Greedy Algorithm for selecting the best seed set of a user-specified budget.
    Assumes simulation looping and averaging are handled by the experiment script.

    Parameters:
        - network: A NetworkX graph object.
        - budget: Number of seed nodes to select.
        - diffusion_model: The diffusion model ("independent_cascade", "linear_threshold", or "pressure_threshold").
        - alpha: Pressure diffusion model parameter (optional, default is 0).

    Returns:
        - best_seed_set: List of selected seed nodes.
        - total_influence: Total influence spread for the selected seed set.
    """

    S, spread, = [], []

    for _ in trange(k):

        max_mg = 0
        for node not in 