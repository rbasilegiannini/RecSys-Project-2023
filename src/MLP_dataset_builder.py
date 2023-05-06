import numpy as np
from numpy.random import default_rng
import utility


def get_training_set(urm, user_item_concatenated_embeddings):
    # Get interaction pairs to put into the training set
    interaction_pairs_tuples = np.where(urm == 1)
    interacting_users = list(zip(interaction_pairs_tuples))[0]
    interacting_items = list(zip(interaction_pairs_tuples))[1]
    # urm[int_users[i], int_items[i]] is an interaction pair

    # Get 4 negative cases for each interaction pair,
    # that are 4 user-item pairs where interaction didn't occurr
    users_number_of_interactions = np.count_nonzero(urm, axis=1)
    users_not_interacting_items = []
    for user in range(urm.shape[0]):
        users_not_interacting_items.append(get_user_negative_cases(user, urm, users_number_of_interactions[user]))

    # Create the training set structure
    number_of_interactions = np.count_nonzero(urm)
    training_set = np.zeros((number_of_interactions + number_of_interactions * 4,
                             user_item_concatenated_embeddings.shape[2] + 1))

    # Iterate and insert interaction pairs in training set with label 1
    for interaction_pair_index in range(number_of_interactions):
        current_interacting_user = interacting_users[0][interaction_pair_index]
        current_interacting_item = interacting_items[0][interaction_pair_index]

        training_set[interaction_pair_index, :user_item_concatenated_embeddings.shape[2]] = \
            user_item_concatenated_embeddings[current_interacting_user, current_interacting_item]
        # training_set[interaction_pair_index, :20] = np.repeat(interaction_pair_index, 20)
        training_set[interaction_pair_index, -1] = 1

    # Iterate and insert negative cases in training set with label 0
    user_index = 0
    negative_cases_base_index = number_of_interactions
    for user_index in range(urm.shape[0]):
        user_negative_cases = users_not_interacting_items[user_index]
        user_negative_cases_embeddings = np.take(
            user_item_concatenated_embeddings[user_index], user_negative_cases, axis=0)

        training_set[negative_cases_base_index:
                     negative_cases_base_index + user_negative_cases_embeddings.shape[0],
                     :user_negative_cases_embeddings.shape[1]] = \
                    user_negative_cases_embeddings

        user_index += 1
        negative_cases_base_index = negative_cases_base_index + user_negative_cases_embeddings.shape[0]

    return training_set


def get_user_negative_cases(interacting_user, urm, number_of_interactions):
    not_interacted_items_tuples = np.where(urm[interacting_user] == 0)
    not_interacted_items = list(zip(not_interacted_items_tuples))

    negative_cases_factor = 4
    while negative_cases_factor * number_of_interactions > not_interacted_items[0][0].shape[0]:
        negative_cases_factor -= 1

    user_negative_cases = np.random.choice(not_interacted_items[0][0],
                                           negative_cases_factor * number_of_interactions,
                                           replace=False)

    return user_negative_cases


if __name__ == "__main__":
    urm = np.identity(10, dtype=int)
    urm[0][3] = 1
    urm[4][7] = 1
    get_training_set(urm, None)