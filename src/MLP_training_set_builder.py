import numpy as np


def get_training_set(urm, user_item_concatenated_embeddings, test_items, l_ext=-0.5, r_ext=0.5):
    # Get interaction pairs to put into the training set
    interacting_users, interacting_items = _get_interaction_pairs(urm)
    # urm[int_users[i], int_items[i]] is an interaction pair

    # Get 4 negative cases for each interaction pair,
    # that are 4 user-item pairs where interaction didn't occur
    users_negative_cases = _get_negative_cases(urm, test_items)

    number_of_negative_cases = _get_number_of_negative_cases(users_negative_cases)

    # Create the training set structure
    number_of_interactions = np.count_nonzero(urm)
    training_set = np.zeros((number_of_interactions + number_of_negative_cases,
                             user_item_concatenated_embeddings.shape[2] + 1))

    # Iterate and insert interaction pairs in training set with label 1
    training_set = _insert_interaction_embeddings_in_training_set(
        training_set, interacting_users, interacting_items, number_of_interactions, user_item_concatenated_embeddings)

    # Iterate and insert negative cases in training set with label 0
    training_set = _insert_negative_cases_embeddings_in_training_set(training_set, users_negative_cases, number_of_interactions,
                                                                     urm.shape[0], user_item_concatenated_embeddings)

    training_set = _normalize_training_set(training_set, r_ext, l_ext)

    np.random.shuffle(training_set)

    return training_set


def _get_interaction_pairs(urm):
    interaction_pairs_tuples = np.where(urm == 1)
    interacting_users = list(zip(interaction_pairs_tuples))[0]
    interacting_items = list(zip(interaction_pairs_tuples))[1]
    return interacting_users, interacting_items


def _get_negative_cases(urm, test_items):
    users_number_of_interactions = np.count_nonzero(urm, axis=1)
    users_negative_cases = []
    for user in range(urm.shape[0]):
        users_negative_cases.append(_get_user_negative_cases(user, urm, users_number_of_interactions[user], test_items[user]))

    return users_negative_cases


def _get_user_negative_cases(interacting_user, urm, number_of_interactions, test_item):
    not_interacted_items_tuples = np.where(urm[interacting_user] == 0)
    not_interacted_items = list(not_interacted_items_tuples[0])

    # Remove test item if it is present
    not_interacted_items.remove(test_item)

    # To avoid overflow when the number of not interacted items is less than negative cases
    negative_cases_factor = 4
    while negative_cases_factor * number_of_interactions > len(not_interacted_items):
        negative_cases_factor -= 1

    user_negative_cases = np.random.choice(not_interacted_items,
                                           negative_cases_factor * number_of_interactions,
                                           replace=False)

    return user_negative_cases


def _get_number_of_negative_cases(users_negative_cases):
    number_of_negative_cases = 0
    for user_negative_cases in users_negative_cases:
        number_of_negative_cases += user_negative_cases.shape[0]
    return number_of_negative_cases


def _insert_interaction_embeddings_in_training_set(training_set, interacting_users, interacting_items,
                                                   number_of_interactions, user_item_concatenated_embeddings):
    for interaction_pair_index in range(number_of_interactions):
        current_interacting_user = interacting_users[0][interaction_pair_index]
        current_interacting_item = interacting_items[0][interaction_pair_index]

        training_set[interaction_pair_index, :user_item_concatenated_embeddings.shape[2]] = \
            user_item_concatenated_embeddings[current_interacting_user, current_interacting_item]
        training_set[interaction_pair_index, -1] = 1

    return training_set


def _insert_negative_cases_embeddings_in_training_set(training_set, users_negative_cases, number_of_interactions,
                                                      number_of_users, user_item_concatenated_embeddings):
    negative_cases_base_index = number_of_interactions
    for user_index in range(number_of_users):
        user_negative_cases = users_negative_cases[user_index]
        user_negative_cases_embeddings = np.take(
            user_item_concatenated_embeddings[user_index], user_negative_cases, axis=0)

        training_set[negative_cases_base_index:
                     negative_cases_base_index + user_negative_cases_embeddings.shape[0],
                     :user_negative_cases_embeddings.shape[1]] = user_negative_cases_embeddings

        negative_cases_base_index = negative_cases_base_index + user_negative_cases_embeddings.shape[0]

    return training_set


def _normalize_training_set(training_set, r_ext, l_ext):
    samples_size = training_set.shape[1] - 1
    normalized_samples = _normalize_samples(training_set[:, :samples_size], l_ext, r_ext)
    training_set[:, :samples_size] = normalized_samples

    return training_set


def _normalize_samples(samples, l_ext, r_ext):
    """
    This function is used to normalize each feature of the samples in [l_ext, r_ext]

    :param samples:
        The dataset's samples. This input must be a matrix where each row is a sample and each column is a feature.
    :param l_ext:
        left end of the interval.
    :param r_ext:
        Right end of the interval.
    :return:
        The normalized samples.
    """

    num_samples = samples.shape[0]
    max_value = samples.max()
    min_value = samples.min()

    for i in range(num_samples):
        samples[i] = l_ext + (((samples[i] - min_value) * (r_ext - l_ext)) / (max_value - min_value))

    return samples
