import networkx as nx
import numpy as np


def generate_bipartite_network(urm):
    """
    :param urm:
        The User-Rating Matrix
    :return:
        A bipartite graph
    """
    _B = nx.Graph()

    # initialize interaction dct with user nodes
    _interaction_dct = {}
    for _user in range(urm.shape[0]):
        _interaction_dct[_user] = []
        _B.add_node(_user, bipartite=0)  # Users set

    # An offset to distinguish users from items id
    offset = urm.shape[0]

    for _item in range(urm.shape[1]):
        _B.add_node(_item + offset, bipartite=1)  # Items set

    # Retrieve user-items interaction
    for _user, _items in _interaction_dct.items():
        _interaction_dct[_user] = np.array(np.where(urm[_user] == 1)).flatten()
        _interaction_dct[_user] += offset

        # Add edges only between nodes of opposite node sets
        for _item in _interaction_dct[_user]:
            _B.add_edge(_user, _item)

    if nx.is_bipartite(_B):
        return _B
