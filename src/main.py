import numpy as np
import neighborhood_builder as nb
import networkx as nx
import matplotlib.pyplot as plt
import scipy.sparse as ss

if __name__ == '__main__':

    URM = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 0, 0, 1, 1, 1]
    ])

    urm_rand = np.random.randint(2, size=(30, 30))
    urm_identity = np.identity(30)

    sparse_matrix = ss.random(943, 1682, density=0.063, format='csr',
                     data_rvs=np.ones,  # fill with ones
                     dtype='f'  # use float32 first
                     ).astype('int8')  # then convert to int8
    urm_sparse = sparse_matrix.toarray()

    B = nb.generate_bipartite_network(URM)

    users = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    items = set(B) - users

    # colors
    c = []

    for node in users:
        c.append('green')

    for node in items:
        c.append('red')

    pos = nx.bipartite_layout(B, users)

    nx.draw(B, pos, node_color = c, with_labels=True, node_size = 100, alpha = 0.6)
    plt.show()

    neighborhoods = nb.extract_neighborhood(URM)
    print(neighborhoods[0])
    print(neighborhoods[1])
