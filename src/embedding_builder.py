import numpy as np

class Embedding:
    def __init__(self, urm):
        [u, s, vh] = np.linalg.svd(urm)

        # compute U and V
        self.__U = u
        self.__V_T = np.diag(s) @ vh

    def get_user_embeddings(self):
        return self.__U

    def get_item_embeddings(self):
        return self.__V_T



