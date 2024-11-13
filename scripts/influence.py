#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 21:39:24 2018

@author: abhishek.umrawal
"""

"importing required built-in modules"
import networkx as nx
import numpy as np

"importing required user-defined modules"
from diffusion_models.independent_cascade import independent_cascade
from diffusion_models.linear_threshold import linear_threshold
from diffusion_models.pressure_diffusion import pressure_linear_threshold

def influence(network, seed_set, diffusion_model, alpha=0):
    
    nodes = list(nx.nodes(network))

    if diffusion_model == "independent_cascade":
        layers = independent_cascade(network, seed_set)  
        
    elif diffusion_model == "linear_threshold":
        layers = linear_threshold(network, seed_set)

    elif diffusion_model == "pressure_threshold":
        layers = pressure_linear_threshold(network, seed_set, 0, alpha)        

    influence = np.sum([len(item) for item in layers])

    return influence