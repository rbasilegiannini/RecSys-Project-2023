import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def generate_network(n):
    '''
    :param n:
        The number of net's nodes
    :return:
        A networkX graph
    '''

    # initialize dictionary with nodes
    _graph_dct = {node: [] for node in range(n)}    # (node, edges[])
    _nodes = list(range(n))     # [1, ..., n]

    # generate edges. For each (node, edges[]), generate a random collection of edges
    for _n, _edge_list in _graph_dct.items():
        _edge_c = random.randint(min(_nodes), int(max(_nodes) / 2)) # edges' cardinality
        _el = random.sample(_nodes, _edge_c)
        _graph_dct[_n] = _el

    # create networkx multi-edge graph
    _G = nx.MultiGraph(_graph_dct)
    return _G

n = 20
G = generate_network(n)

# visualize graph
pos = nx.spring_layout(G)
nx.draw(G, pos, node_size = 75, alpha = 0.8)
plt.show()
