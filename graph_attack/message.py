#import dgl
import numpy as np
import torch
import networkx as nx
import random
import math
import time

def genRandomGraphX2(num_nodes, num_edges):
    #num_nodes -= 1 # the graph is zero-indexed but the total number at the end isn't?!
    G = nx.cycle_graph(num_nodes)

    # random edges:
    edges_added = 0
    while edges_added < num_edges-num_nodes:
        u,v = random.randint(0,num_nodes), random.randint(0,num_nodes)
        if u != v:
            if not G.has_edge(u,v):
                G.add_edge(u,v)
                edges_added += 1

    pos = nx.kamada_kawai_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=[[.7, .7, .7]])
    print("edges:", G.edges())
    return G

def efficiencyMetric(G):
    num_nodes = len(G.nodes())
    floyd = nx.floyd_warshall_numpy(G)
    tri_floyd = np.triu(floyd, k=1)
    
    tri_floyd = np.clip(tri_floyd, 0, 1000)
    
    num_terms = ((num_nodes**2)-num_nodes)/2
    _mean = np.sum(tri_floyd)/num_terms
    assert _mean != np.inf
    return _mean

def using_indexed_assignment(x):
    result = np.empty(len(x), dtype=int)
    temp = x.argsort()
    result[temp] = np.arange(len(x))
    return result

def get_y(G, v):
    
    eff_list = list()
    num_nodes = G.number_of_nodes()
    
    for u in range(num_nodes):
        
        if not G.has_edge(u,v):
            G.add_edge(u,v)
            # calculate effiency right here:
            score = efficiencyMetric(G)
            eff_list.append(score)
            G.remove_edge(u,v)
        else:
            score = efficiencyMetric(G)
            eff_list.append(score)
                    
    _sorted = using_indexed_assignment(np.array(eff_list))
    
    print(eff_list)
    
    return (_sorted / _sorted.max()) ** 3
    
    
def Generate_dataset(N):
    
    out = []
    for i in range(N):
        G = genRandomGraphX2(20,30)
        V = np.random.randint(20)
        X = np.ones(G.number_of_nodes() * 2).reshape(G.number_of_nodes(), 2)

        X[:, 1] = 0
        X[V, 0] = 0
        X[V, 1] = 1

        Y = get_y(G, V)
        
        out.append([G, X, Y])

    return out
    