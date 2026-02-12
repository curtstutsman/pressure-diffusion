"""
Experiment: Evaluating True Influence Spread of Different IM-Solvers under Various Diffusion Models

Author: Curt Stutsman
Date: 2/4/2025

Description:
    The purpose of this script is to compare the performance of different heuristics and IM-Solving algorithms under 
    under the pressure threshold and linear threshold diffusion model. Each algorithm will have its influence evaluated on the facebook SNAPS sample graph (4,039 nodes, 88,234 edges)
    with budgets ranging (0, 20]

    Graphs Evaluated:
    1. SNAP's Facebook: https://snap.stanford.edu/data/ego-Facebook.html

Goals:
- Simulate diffusion for multiple IM solving algs/heuristics
- Compare performance as budget increases

Inputs:
--alpha:        Pressure parameter alpha to control influence propagation.
--simulations:  Number of simulations to run per heuristic.

Outputs:
- Results are written to a CSV file after simulations.
"""

#=======================================
#   Library Imports
#======================================
import numpy as np
import pandas as pd
import os
from src.scripts import celf
from src.utils.graph_utils import create_facebook_graph
from cynetdiff.utils import networkx_to_pt_model, networkx_to_lt_model
from cynetdiff.models import DiffusionModel
from src.utils.path_utils import ROOT

# Global Constants
MODEL = 'pressure_threshold'
OUTPUT_DIR = ROOT + "/data/results"

#=======================================
#   Main Execution
#=======================================
if __name__ == "__main__":
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(OUTPUT_DIR, "main.csv")
    if os.path.exists(results_file):
        os.remove(results_file)

    pd.DataFrame(columns=["Threshold Distribution", "Alpha", "Average Influence"]).to_csv(results_file, index=False)
    
    models = []
    n=4039
    fb = create_facebook_graph()
    trunc_normal = np.ascontiguousarray(np.clip(np.random.normal(0.5, 0.15, n), 0, 1).astype(np.float32))
    beta_low = np.ascontiguousarray(np.random.beta(2, 5, n).astype(np.float32))
    beta_high = np.ascontiguousarray(np.random.beta(5, 2, n).astype(np.float32))

    models.append((networkx_to_lt_model(fb)[0], "Truncated Normal", "0.000"))
    models.append((networkx_to_pt_model(fb, 0.001)[0], "Truncated Normal", '0.001'))
    models.append((networkx_to_pt_model(fb, 0.005)[0], "Truncated Normal", "0.005"))
    models[0][0]._assign_thresholds(trunc_normal)
    models[1][0]._assign_thresholds(trunc_normal)
    models[2][0]._assign_thresholds(trunc_normal)

    models.append((networkx_to_lt_model(fb)[0], "Low-Threshold Beta", "0.000"))
    models.append((networkx_to_pt_model(fb, 0.001)[0], "Low-Threshold Beta", '0.001'))
    models.append((networkx_to_pt_model(fb, 0.005)[0], "Low-Threshold Beta", "0.005"))
    models[3][0]._assign_thresholds(beta_low)
    models[4][0]._assign_thresholds(beta_low)
    models[5][0]._assign_thresholds(beta_low)


    models.append((networkx_to_lt_model(fb)[0], "High-Threshold Beta", "0.000"))
    models.append((networkx_to_pt_model(fb, 0.001)[0], "High-Threshold Beta", '0.001'))
    models.append((networkx_to_pt_model(fb, 0.005)[0], "High-Threshold Beta", "0.005"))
    models[6][0]._assign_thresholds(beta_high)
    models[7][0]._assign_thresholds(beta_high)
    models[8][0]._assign_thresholds(beta_high)

    for model in models:
        seeds, spread = celf.celf(model[0], n, 61)
        new_result = {
            "Threshold Distribution" : model[1],
            "Alpha" : model[2],
            "Average Influence" : spread, 
        }
        results_df = pd.DataFrame([new_result])
        results_df.to_csv(results_file, mode='a', header=False, index=False)
