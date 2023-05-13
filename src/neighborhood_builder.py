import random
import networkx as nx
import numpy as np


def generate_bipartite_network(urm):
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
    [bi_graph, offset] = generate_bipartite_network(urm)
    communities = nx.algorithms.community.louvain_communities(bi_graph, resolution=resolution)

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
                    neighborhood = handle_empty_neighborhood(user_interactions)

                users_neighborhood[user] = neighborhood

            else:  # node is an item
                item = node - offset
                # Remove other items from the community
                for n in neighborhood:
                    if n in item_nodes:
                        neighborhood = np.delete(neighborhood, np.where(neighborhood == n))

                # Collect the item's neighborhood (of users)
                if neighborhood.size == 0:
                    item_interactions = urm[:, item]
                    neighborhood = handle_empty_neighborhood(item_interactions)

                items_neighborhood[item] = neighborhood

    return [users_neighborhood, items_neighborhood]


def handle_empty_neighborhood(node_interactions):
    """
    Empty neighborhood handler. Fill the neighborhood with direct node interactors.
    If there aren't direct node interactors the neighborhood will contain only one random neighbor.
    :param node_interactions:
        Node's direct interactions.
    :return:
        Neighborhood filled.
    """

    if max(node_interactions) == 0:
        random_user = random.randint(0, node_interactions.size)
        user_interacted = np.array([random_user])
    else:
        user_interacted = np.where(node_interactions == 1)

    neighborhood = np.array(user_interacted)

    return neighborhood
