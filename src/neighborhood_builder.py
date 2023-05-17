import random
import networkx as nx
import numpy as np


def extract_neighborhood(urm, resolution=0.5):
    """
    :param urm:
        The User-Rating Matrix
    :param resolution:
        This value is between [0, 1].
        If resolution is less than 1, the algorithm favors larger communities.
    :return:
        A list with users neighborhood (index 0) and items neighborhood (index 1)
    """

    # Check resolution
    if resolution > 1 or resolution < 0:
        resolution = 0.5

    # Mapping resolution in [1, 10] because of louvain_communities input
    input_min = 0
    input_max = 1
    res_min = 1
    res_max = 10
    slope = (res_max - res_min) / (input_max - input_min)
    resolution = res_min + slope * (resolution - input_min)

    # extract communities from URM
    [bi_graph, offset] = _generate_bipartite_network(urm)
    communities = nx.algorithms.community.louvain_communities(bi_graph, resolution=resolution, seed=0)

    user_nodes = {n for n, d in bi_graph.nodes(data=True) if d["bipartite"] == 0}
    item_nodes = set(bi_graph) - user_nodes

    users_neighborhood = np.empty(shape=len(user_nodes), dtype=np.ndarray)
    items_neighborhood = np.empty(shape=len(item_nodes), dtype=np.ndarray)

    # for each community, retrieve the user's (item's) neighborhood
    for community in communities:

        for node in community:
            neighborhood = np.array(list(community))

            if node in user_nodes:  # node is a user
                user = node
                # Remove other users from the community
                for n in neighborhood:
                    if n in user_nodes:
                        neighborhood = np.delete(neighborhood, np.where(neighborhood == n))
                neighborhood = neighborhood - offset

                # Collect the user's neighborhood (of items)
                if neighborhood.size == 0:
                    user_interactions = urm[user, :]
                    neighborhood = _handle_empty_neighborhood(user_interactions)

                users_neighborhood[user] = neighborhood

            else:  # node is an item
                item = node - offset
                # Remove other items from the community
                for n in neighborhood:
                    if n in item_nodes:
                        neighborhood = np.delete(neighborhood, np.where(neighborhood == n))

                # Collect the item's neighborhood (of users)
                if neighborhood.size == 0:
                    urm_item_column = urm[:, item]
                    neighborhood = _handle_empty_neighborhood(urm_item_column)

                items_neighborhood[item] = neighborhood

    # Cut neighborhoods
    for user in range(len(user_nodes)):
        if users_neighborhood[user].size > 50:
            users_neighborhood[user] = users_neighborhood[user][:50]

    for item in range(len(item_nodes)):
        if items_neighborhood[item].size > 50:
            items_neighborhood[item] = items_neighborhood[item][:50]

    return [users_neighborhood, items_neighborhood]


def _generate_bipartite_network(urm):
    """
    :param urm:
        The User-Rating Matrix
    :return:
        A bipartite graph and the offset between user's and item's ID
    """

    bi_graph = nx.Graph()

    # initialize interaction dct with user nodes
    interaction_dct = {}
    for user in range(urm.shape[0]):
        interaction_dct[user] = []
        bi_graph.add_node(user, bipartite=0)  # Users set

    # An offset to distinguish users from items id
    offset = urm.shape[0]

    for item in range(urm.shape[1]):
        bi_graph.add_node(item + offset, bipartite=1)  # Items set

    # Retrieve user-items interaction
    for user, items in interaction_dct.items():
        interaction_dct[user] = np.array(np.where(urm[user] == 1)).flatten()
        interaction_dct[user] += offset

        # Add edges only between nodes of opposite node sets
        for item in interaction_dct[user]:
            bi_graph.add_edge(user, item)

    if nx.is_bipartite(bi_graph):
        return [bi_graph, offset]


def _handle_empty_neighborhood(urm_node_column):
    """
    Empty neighborhood handler. Fill the neighborhood with direct node interactors.
    If there aren't direct node interactors the neighborhood will contain only one random neighbor.
    :param urm_node_column:
        Urm information about the node.
    :return:
        Neighborhood filled.
    """

    if max(urm_node_column) == 0:
        random_node = random.randint(0, urm_node_column.size)

    else:
        interaction_nodes_tuple = np.where(urm_node_column == 1)
        interaction_nodes_array = interaction_nodes_tuple[0]
        random_node = np.random.choice(interaction_nodes_array)

    neighborhood = np.array([random_node])

    return neighborhood
