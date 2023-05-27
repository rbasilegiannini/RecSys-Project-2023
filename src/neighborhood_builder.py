import random
import networkx as nx
import numpy as np
from community import community_louvain


def extract_urm_neighborhoods(urm, resolution=0.5):
    """
    This function extracts all neighborhoods from the URM.
    :param urm:
        The User-Rating Matrix
    :param resolution:
        This value is between [0, 1].
        If resolution is less than 1, the algorithm favors larger communities.
    :return:
        A list with users neighborhood (index 0) and items neighborhood (index 1)
    """

    # Check resolution
    if resolution > 1:
        resolution = 1
    if resolution < 0:
        resolution = 0

    # extract communities from URM
    [bi_graph, offset] = _generate_bipartite_network(urm)
    communities = community_louvain.best_partition(bi_graph, resolution=resolution)

    user_nodes = {n for n, d in bi_graph.nodes(data=True) if d["bipartite"] == 0}
    item_nodes = set(bi_graph) - user_nodes

    users_neighborhood = np.empty(shape=len(user_nodes), dtype=np.ndarray)
    items_neighborhood = np.empty(shape=len(item_nodes), dtype=np.ndarray)

    # From key = 0 to key = 942 are users. The values represent the community's id
    communities_with_users = {node: communities[node] for node in range(len(user_nodes))}
    communities_with_items = {node: communities[node] for node in range(offset, offset + len(item_nodes))}

    for node, community_id in communities.items():

        if node in user_nodes:  # node is a user
            user = node
            # user_neighborhood_list = []
            # for neighbor, neighborhood_id in communities_with_items.items():
            #     if neighborhood_id == community_id:
            #         user_neighborhood_list.append(neighbor)
            # user_neighborhood = np.array(user_neighborhood_list)
            # user_neighborhood = user_neighborhood - offset
            #
            # # Check neighborhood size
            # if user_neighborhood.size > 50:
            #     user_neighborhood = user_neighborhood[:50]
            #
            # elif user_neighborhood.size == 0:
            #     urm_user_row = urm[user, :]
            #     user_neighborhood = _handle_empty_neighborhood(urm_user_row)
            #
            # users_neighborhood[user] = user_neighborhood

            users_neighborhood[user] = _generate_node_neighborhood(user,
                                                                   communities_with_items,
                                                                   community_id,
                                                                   urm,
                                                                   offset,
                                                                   is_user=True
                                                                   )

        else:  # node is an item
            item = node - offset
            # item_neighborhood_list = []
            # for neighbor, neighborhood_id in communities_with_users.items():
            #     if neighborhood_id == community_id:
            #         item_neighborhood_list.append(neighbor)
            # item_neighborhood = np.array(item_neighborhood_list)
            #
            # # Check neighborhood size
            # if item_neighborhood.size > 50:
            #     item_neighborhood = item_neighborhood[:50]
            # elif item_neighborhood.size == 0:
            #     urm_item_column = urm[:, item]
            #     item_neighborhood = _handle_empty_neighborhood(urm_item_column)
            #
            # items_neighborhood[item] = item_neighborhood
            items_neighborhood[item] = _generate_node_neighborhood(item,
                                                                   communities_with_users,
                                                                   community_id,
                                                                   urm,
                                                                   offset,
                                                                   is_user=False
                                                                   )

    return [users_neighborhood, items_neighborhood]


def _generate_bipartite_network(urm):
    """
    This function generates a bipartite graph from an urm.
    :param urm:
        The User-Rating Matrix.
    :return:
        A bipartite graph and the offset between user's and item's ID.
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
    Empty neighborhood handler. Fill the neighborhood with one direct interactor node.
    If there aren't direct interactor nodes the neighborhood will contain only one random neighbor.
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


def _generate_node_neighborhood(node, communities, community_id, urm, offset, is_user):
    """
    This function extracts the node's neighborhood.

    :param node:
        The node of interest (it's an ID).
    :param communities:
        The communities-dictionary (the items will be like [node_id, neighborhood_id]).
    :param community_id:
        The community's id of the node (each node has a community).
    :param urm:
        The urm.
    :param offset:
        The offset between user_id and item_id.
    :param is_user:
        A boolean value (True for user, False for item).
    :return:
        The node's neighborhood.
    """

    node_neighborhood_list = []
    for neighbor, neighborhood_id in communities.items():
        if neighborhood_id == community_id:
            node_neighborhood_list.append(neighbor)
    node_neighborhood = np.array(node_neighborhood_list)
    if is_user:
        node_neighborhood = node_neighborhood - offset

    # Check neighborhood size
    if node_neighborhood.size > 50:
        node_neighborhood = node_neighborhood[:50]

    elif node_neighborhood.size == 0:
        if is_user:
            urm_user_row = urm[node, :]
            node_neighborhood = _handle_empty_neighborhood(urm_user_row)
        else:
            urm_item_column = urm[:, node]
            node_neighborhood = _handle_empty_neighborhood(urm_item_column)

    return node_neighborhood
