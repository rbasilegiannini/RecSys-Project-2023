import numpy as np


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
