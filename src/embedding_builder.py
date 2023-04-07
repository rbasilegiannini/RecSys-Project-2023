import numpy as np
from keras.layers import Conv1D, MaxPooling1D


# TODO: decide if class is needed
class Embedding:
    def __init__(self, urm, latent_factor_size):
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

        # Compute U and V
        self.__U = u
        self.__V = np.transpose(s @ vh)

    def get_user_embeddings(self):
        """
        :return:
            Matrix of user latent vectors
        """
        return self.__U

    def get_item_embeddings(self):
        """
        :return:
            Matrix of item latent vectors (transpose)
        """
        return self.__V

    def get_interaction_functions(
            self, users_latent_vector, items_latent_vector, latent_factor_size):
        interaction_functions = np.zeros(
            (users_latent_vector.shape[0], items_latent_vector.shape[0], latent_factor_size))

        for user in range(users_latent_vector.shape[0]):
            for item in range(items_latent_vector.shape[0]):
                interaction_functions[user, item, :] = users_latent_vector[user, :] * items_latent_vector[item, :]

        return interaction_functions

    def get_neighborhoods_embedding(self, neighborhoods_encoding, embedding):
        # First, we get p_u(N) (or q_i(N)), using a projection of "embeddings"
        # through "neighborhoods_encoding"
        normalized_neighborhoods_embedding = []
        for neighborhood_encoding in neighborhoods_encoding:
            neighborhood_embedding = embedding[np.where(neighborhood_encoding == 1)]
            # Then, we normalize p_u(N) (or q_i(N))
            normalized_neighborhood_embedding = self.normalize_neighborhood_embedding(neighborhood_embedding)
            normalized_neighborhoods_embedding.append(normalized_neighborhood_embedding)

        return normalized_neighborhoods_embedding

    def normalize_neighborhood_embedding(self, neighborhood_embedding):
        # Convolute the neighborhood latent vectors and max-pool it
        reshaped_neighborhood_embedding = neighborhood_embedding.reshape(
            neighborhood_embedding.shape[0], neighborhood_embedding.shape[1], 1)

        # |N| x k x |kernels|
        convoluted_neighborhood_embedding = Conv1D(32, 5, activation='relu',
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
