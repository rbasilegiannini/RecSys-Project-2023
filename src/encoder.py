import numpy as np


def get_onehot_encoding(size):
    onehot_matrix = np.zeros((size, size), dtype=int)
    for i in range(size):
        onehot_matrix[i][i] = 1

    return onehot_matrix
