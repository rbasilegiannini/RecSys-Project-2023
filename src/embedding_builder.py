import numpy as np
import keras
from keras.layers import Conv1D,MaxPooling1D

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
        self.__V_T = s @ vh

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
        return self.__V_T

    def get_neighborhood_embeddings(self, neighborhoods_encoding, embeddings):
        # First, we get p_u(N) and q_i(N), using a projection of "embeddings"
        # through "neighborhoods_encoding"
        # Then, we normalize p_u(N) and q_i(N)
        pass

    def normalize_embeddings(self, neighborhoods_embeddings):
        # For each neighborhood, convolute its latent vectors and max-pool it
        for neighborhood_embeddings in neighborhoods_embeddings:
            reshaped_neighborhood_embeddings = neighborhood_embeddings.reshape(
                neighborhood_embeddings.shape[0], neighborhood_embeddings.shape[1], 1)

            convoluted_neighborhood_embeddings = Conv1D(32, 5, activation='relu',
                                                        padding='same',
                                                        input_shape=reshaped_neighborhood_embeddings[1:]
                                                        )(reshaped_neighborhood_embeddings)
            pooled_neighborhood_embeddings = MaxPooling1D(2)(convoluted_neighborhood_embeddings)

            # To set a fixed dimension for the neighborhood latent vectors,
            # indipendent from the neighborhood size,
            # given a latent factor and a kernel, compute the average of the kernel on the neighbors
            latent_factors_size = pooled_neighborhood_embeddings.shape[1]
            kernels_size = pooled_neighborhood_embeddings.shape[2]
            neighborhood_embeddings_averaged_by_kernel = np.zeros((latent_factors_size, kernels_size))

            for factor_index in latent_factors_size:
                for kernel_index in kernels_size:
                    neighborhood_embeddings_averaged_by_kernel[factor_index, kernel_index] = np.average(
                        pooled_neighborhood_embeddings[:, factor_index, kernel_index])

            return neighborhood_embeddings_averaged_by_kernel
