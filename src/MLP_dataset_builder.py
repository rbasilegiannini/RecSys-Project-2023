import numpy as np
from numpy.random import default_rng


def get_training_set(urm, user_item_concatenated_embeddings):
    # Get interaction pairs to put into the training set
    interaction_pairs_tuples = np.where(urm == 1)
    users_interaction_pairs = list(zip(interaction_pairs_tuples))[0]
    items_interaction_pairs = list(zip(interaction_pairs_tuples))[1]
    # urm[user_int_pairs[i], items_int_pairs[i]] is an interaction pair


    # Get 4 negative cases for each interaction pair,
    # that are 4 user-item pairs where interaction not occurred
    users_number_of_interactions = np.count_nonzero(urm, axis=1)
    users_not_interacting_items = []
    for user in range(urm.shape[0]):
        users_not_interacting_items.append(get_user_negative_cases(user, urm))

    # Build the training set with the retrieved pairs


def get_user_negative_cases(interacting_user, urm):
    not_interacted_items_tuples = np.where(urm[interacting_user] == 0)
    not_interacted_items = list(zip(not_interacted_items_tuples))
    user_negative_cases = np.random.choice(not_interacted_items[0][0], 4, replace=False)

    return user_negative_cases


