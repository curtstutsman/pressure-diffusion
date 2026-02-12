# pressure-diffusion
1. Comparison of heuristics (using influence and true influence), all heuristics listed in /data/resultsheuristic.txt

2. Basic demonstration of PLT on a small graph compared to LT

3. Run various IM algs on PLT and LT and compare outputs
    a. Greedy
    b. CELF++

4. Test greedy on LT and PT and see if they select same seeds

5. Types of graphs to test (try for all experiments)
    a. Random graphs (current)
    b. Graph with nodes w/ more parents than children on average
    c. Louvaine sample of Facebook graph
    d. stochastic block model

- How does new diffusion model change the influence maximization problem
- Need to rethink when we update edge weights. Should we only do it on activation? Should we update the edge weights of all active nodes during each timestep?
- How should I talk about my work on the CyNetDiff library in the thesis?
- Would be nice to see a different way of testing. I want to try predicting the spread of a historic diffusion, and seeing which model does a better job at matching the final spread. Could also apply statistical tests to compare predicted cascade curbes with actual cascase curves. 
    If the PT model has lower error rates than the LT model, it proves its relevance. 
 
There are a whole new layer of considerations with this new diffusion model that I did not previously think about. Currently, a node's outbound influence is updated upon activation and only updated for edges connected to nodes that were activated in a previous round. 
But should the edge be updated on activation or should the updates all occur simultaneously at the end? If we allow updates on edges to nodes that will later become activated in the same time step, then the processing order of nodes effects the diffusion. 
But we have no way of knowing if a node will later become activated in the same time step. The ideal solution is then to track all nodes activated in a time step, and at the end of the activation phase, begin an an update phase where we update the outbound edges of all newly activated nodes, 
making sure not to adjust edges towards already activated nodes AND newly activated nodes. This ensures that processing order of nodes does not impact diffusion. It is unclear if neighboring newly activated nodes should boost each others outbound influence to non active nodes

THINGS TO TEST:
- Track underlying set and compare differences (include smaller examples as well)
- Write theorem that influence from PT model is never less than influence from LT modelm (a = 0 gives LT model)
    - Run LT and PT with varying alpha on the exact same network with the exact same seed nodes and save spreads
    - Looking to find f(a) = {s(PT) - s(LT)}
- Looking to convince people ot use my model
- Erdosh Reny Random Graphs, Bitcoin, Wikipedia, Epinions, Community based graphs, graphs w certain number of parents
- Try also scaling up graph while keeping edge probability and alpha constant
- Try also scaling up edge probability while keeping alpha and size constant
- Try also mapping the spread of the graph as a function of the size of the seed set


Graph loaded from pickle file.
Number of nodes: 4039
Number of edges: 88234
Graph loaded from pickle file.
Number of nodes: 7115
Number of edges: 100762
Graph loaded from pickle file.
Number of nodes: 5881
Number of edges: 35592
Number of nodes: 5000
Number of edges: 62597

varying alpha graph uses centered sliding window average w/ window = 9