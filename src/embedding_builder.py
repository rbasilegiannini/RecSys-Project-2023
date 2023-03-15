import numpy as np

class Embedding:
    def __init__(self, urm, latent_factor_size):
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
        return self.__U

    def get_item_embeddings(self):
        return self.__V_T



