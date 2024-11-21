#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:29:52 2018

@author: abhishek.umrawal
"""
from diffusion_models.independent_cascade import independent_cascade
from diffusion_models.linear_threshold import linear_threshold
from diffusion_models.pressure_diffusion import pressure_linear_threshold
from scripts.weighted_network import weighted_network
import networkx as nx
import numpy as np

def greedy_im(network, budget, diffusion_model, method="rn", alpha=0):
    
    """
    Greedy Algorithm for finding the best seed set of a user-specified budget
    
    Inputs:
        - network is a networkx object
        - budget is the user-specified marketing budget which represents the no.
          of individuals to be given the freebies
        - diffusion model is either "independent_cascade" or "linear_threshold"
        - spontaneous_prob is a vector of spontaneous adoption probabiities for
          each node
          
    Outputs:
        - best_seed_set is a subset of the set of nodes with the cardinality  
          same as the budget such that it maximizes the spread of marketing
        - max_influence is the value of maximum influence
  
    """
    J = 10
    nodes = list(nx.nodes(network))
    max_influence = []
    best_seed_set = []

    for l in range(budget):
        nodes_to_try = list(set(nodes)-set(best_seed_set))
        influence = np.zeros(len(nodes_to_try))
        
        for i in range(len(nodes_to_try)):
            
            for j in range(J):
                
                best_seed_set_plus_ith_node = \
                    list(set(best_seed_set + [nodes_to_try[i]]))
                
                if diffusion_model == "independent_cascade":
                    layers = independent_cascade(network, best_seed_set_plus_ith_node)  
                
                elif diffusion_model == "linear_threshold":
                    layers = linear_threshold(network, best_seed_set_plus_ith_node)    

                elif diffusion_model == "pressure_threshold":
                    layers = pressure_linear_threshold(network, best_seed_set_plus_ith_node, alpha=alpha)
                
                for k in range(len(layers)):
                    influence[i] = influence[i] + len(layers[k])
                    
            influence[i] = influence[i]/J
        
        max_influence.append(np.max(influence))    
        best_seed_set.append(nodes_to_try[np.argmax(influence)])
    
    
    return best_seed_set, max_influence