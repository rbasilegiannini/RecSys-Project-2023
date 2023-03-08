import numpy as np


def get_onehot_encoding(size):
    onehot_matrix = np.identity(size, dtype=int)
    return onehot_matrix
