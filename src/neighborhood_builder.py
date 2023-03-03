import networkx as nx
import numpy as np


def generate_bipartite_network(urm):
    """
    :param urm:
        The User-Rating Matrix
    :return:
        A bipartite graph
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
        return bi_graph


def extract_neighborhood(urm):
    """
    :param urm:
        The User-Rating Matrix
    :return:
        A list with users neighborhood (index 0) and items neighborhood (index 1)
    """

    users_neighborhood = []
    items_neighborhood = []

    # extract communities from URM
    bi_graph = generate_bipartite_network(urm)
    communities = nx.algorithms.community.louvain_communities(bi_graph)

    user_nodes = {n for n, d in bi_graph.nodes(data=True) if d["bipartite"] == 0}
    item_nodes = set(bi_graph) - user_nodes

    # for each community, retrieve the user's (item's) neighborhood
    for community in communities:

        for node in community:
            neighborhood = np.array(list(community))

            if node in user_nodes:  # node is a user
                # Remove other users from the community
                for n in neighborhood:
                    if n in user_nodes:
                        neighborhood = np.delete(neighborhood, np.where(neighborhood == n))

                # Collect the user's neighborhood (of items)
                users_neighborhood.append(neighborhood)

            else:  # node is an item
                # Remove other items from the community
                for n in neighborhood:
                    if n in item_nodes:
                        neighborhood = np.delete(neighborhood, np.where(neighborhood == n))

                # Collect the item's neighborhood (of users)
                items_neighborhood.append(neighborhood)

    return [users_neighborhood, items_neighborhood]
