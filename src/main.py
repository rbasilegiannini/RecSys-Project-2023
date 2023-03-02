import numpy as np
import neighborhood_builder as nb
import networkx as nx
import matplotlib.pyplot as plt

if __name__ == '__main__':

    URM = np.array([
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0]
    ])

    B = nb.generate_bipartite_network(URM)

    top = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}

    pos = nx.bipartite_layout(B, top)
    nx.draw(B, pos, node_size = 75, alpha = 0.8)
    plt.show()