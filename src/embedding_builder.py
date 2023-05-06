import numpy as np
from keras.layers import Conv1D, MaxPooling1D

import utility


def get_latent_vectors(urm, latent_factor_size):
    """
    :param urm:
        The User-Rating Matrix
    :param latent_factor_size:
        The number of latent factors
    """
    [u, s, vh] = np.linalg.svd(urm, full_matrices=False)
    s = np.diag(s)

    # Take most significant features
    u = u[:, 0:latent_factor_size]
    s = s[0:latent_factor_size, 0: latent_factor_size]
    vh = vh[0:latent_factor_size, :]

    return [u, np.transpose(s @ vh)]


def get_interaction_functions(users_latent_vector, items_latent_vector, latent_factor_size):
    interaction_functions = np.zeros(
        (users_latent_vector.shape[0], items_latent_vector.shape[0], latent_factor_size))

    for user in range(users_latent_vector.shape[0]):
        for item in range(items_latent_vector.shape[0]):
            interaction_functions[user, item, :] = users_latent_vector[user, :] * items_latent_vector[item, :]

    return interaction_functions





def get_neighborhoods_embedding(neighborhoods_encoding, embedding):
    # First, we get p_u(N) (or q_i(N)), using a projection of "embeddings"
    # through "neighborhoods_encoding"
    normalized_neighborhoods_embedding = []
    for neighborhood_encoding in neighborhoods_encoding:
        neighborhood_embedding = embedding[np.where(neighborhood_encoding == 1)]
        # Then, we normalize p_u(N) (or q_i(N))
        normalized_neighborhood_embedding = normalize_neighborhood_embedding(neighborhood_embedding)
        normalized_neighborhoods_embedding.append(normalized_neighborhood_embedding)

    # Finally, we convert the neighborhoods embedding in a numpy array
    # (couldn't do before because normalized neighborhood embedding size wasn't known)
    normalized_neighborhoods_embedding_matrix = utility.get_matrix_from_list(
        normalized_neighborhoods_embedding, normalized_neighborhoods_embedding[0].shape[0])
    return normalized_neighborhoods_embedding_matrix


def normalize_neighborhood_embedding(neighborhood_embedding):
    # Convolute the neighborhood latent vectors and max-pool it
    reshaped_neighborhood_embedding = neighborhood_embedding.reshape(
        neighborhood_embedding.shape[0], neighborhood_embedding.shape[1], 1)

    # |N| x k x |kernels|
    convoluted_neighborhood_embedding = Conv1D(2, 5, activation='relu',
                                               padding='same',
                                               input_shape=reshaped_neighborhood_embedding[1:]
                                               )(reshaped_neighborhood_embedding)
    # |N| x k/2 x |kernels|
    pooled_neighborhood_embedding = MaxPooling1D(2)(convoluted_neighborhood_embedding)

    # To set a fixed dimension for the neighborhood latent vectors,
    # independent of the neighborhood size,
    # given a latent factor and a kernel, compute the average of the kernel on the neighbors
    latent_factors_size = pooled_neighborhood_embedding.shape[1]
    kernels_size = pooled_neighborhood_embedding.shape[2]
    averaged_neighborhood_embedding = np.zeros((latent_factors_size, kernels_size))
    for factor_index in range(latent_factors_size):
        for kernel_index in range(kernels_size):
            averaged_neighborhood_embedding[factor_index, kernel_index] = np.average(
                pooled_neighborhood_embedding[:, factor_index, kernel_index])

    # The neighborhood embedding is flattened, since kernel dimension is not relevant
    # for the upper abstraction levels
    averaged_neighborhood_embedding = averaged_neighborhood_embedding.flatten()

    return averaged_neighborhood_embedding


def get_concatenated_embeddings(interaction_functions, users_neighborhood_embedding, items_neighborhood_embedding):
    number_of_users = interaction_functions.shape[0]
    number_of_items = interaction_functions.shape[1]
    embedding_size = interaction_functions.shape[2]
    neighborhood_embedding_size = users_neighborhood_embedding.shape[1]  # Same as items_neighborhood_embedding

    concatenated_embeddings = np.zeros((
        number_of_users,
        number_of_items,
        embedding_size + neighborhood_embedding_size * 2))

    for user in range(number_of_users):
        for item in range(number_of_items):
            concatenated_embeddings[user, item, :] = np.concatenate((
                interaction_functions[user, item],
                users_neighborhood_embedding[user],
                items_neighborhood_embedding[item]))

    return concatenated_embeddings

